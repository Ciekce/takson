use super::simd::*;
use crate::core::{Color, normalizer};
use crate::eval::*;
use crate::position::Position;

const L1_SIZE: usize = 2560;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;

#[cfg(target_feature = "avx512f")]
const UPDATE_CHUNK: usize = L1_SIZE / 4;
#[cfg(not(target_feature = "avx512f"))]
const UPDATE_CHUNK: usize = 256;

const L1_PAIR_COUNT: usize = L1_SIZE / 2;
const L2_PAIR_COUNT: usize = L2_SIZE / 2;

const RESERVE_BUCKET_COUNT: usize = 8;
const KOMI_BUCKET_COUNT: usize = (Position::MAX_HALF_KOMI - Position::MIN_HALF_KOMI) as usize + 1;
const OUTPUT_BUCKET_COUNT: usize = KOMI_BUCKET_COUNT * RESERVE_BUCKET_COUNT;

const Q: i32 = 64;

const Q0_BITS: i32 = 8;
const Q1_BITS: i32 = 7;
const Q_BITS: i32 = 6;

#[repr(C, align(64))]
struct Network {
    ft: FeatureTransformer<L1_SIZE>,
    l1w: [[i8; L1_SIZE / 2 * L2_SIZE]; OUTPUT_BUCKET_COUNT],
    l1b: [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    l2w: [[i16; L2_SIZE * L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l2b: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l3w: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l3b: [i32; OUTPUT_BUCKET_COUNT],
}

static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("EVALFILE"))) };

const SHIFT: i32 = Q0_BITS * 2 + MULHI_SHIFT + Q1_BITS - Q_BITS;

const L2_UNROLL: usize = 8 / (L2_SIZE / CHUNK_SIZE_I32);

#[must_use]
pub fn eval_value(pos: &Position) -> f32 {
    let output_bucket = calc_output_bucket(pos);

    let acc = accumulate(pos);

    let mut activated_ft = Align64([0; L1_PAIR_COUNT]);
    activate_ft::<L1_SIZE, L1_PAIR_COUNT>(&mut activated_ft.0, &acc.0);

    let mut l1_intermediate = [[zero_i32(); L2_SIZE / CHUNK_SIZE_I32]; L2_UNROLL];
    {
        let w = NET.l1w[output_bucket].as_ptr();
        let ft_groups = activated_ft.0.as_ptr();

        for i_outer in (0..L1_PAIR_COUNT / 4).step_by(2 * L2_UNROLL) {
            for (i_inner, intermediate) in l1_intermediate.iter_mut().enumerate() {
                let i_0 = i_outer + 2 * i_inner;
                let i_1 = i_0 + 1;
                let ft_vec_0 = unsafe { set1_u8x4(ft_groups.add(i_0 * 4)) };
                let ft_vec_1 = unsafe { set1_u8x4(ft_groups.add(i_1 * 4)) };

                for (j, intermediate) in intermediate.iter_mut().enumerate() {
                    let weights_0 = unsafe { load_i8(w.add(i_0 * L2_SIZE * 4 + j * CHUNK_SIZE_I8)) };
                    let weights_1 = unsafe { load_i8(w.add(i_1 * L2_SIZE * 4 + j * CHUNK_SIZE_I8)) };
                    *intermediate = dpbusdx2_i32(*intermediate, ft_vec_0, weights_0, ft_vec_1, weights_1);
                }
            }
        }
    }

    let mut l1_out = Align64([0i16; L2_SIZE]);
    let skip;
    {
        let l1_bias = NET.l1b[output_bucket].as_ptr();

        let mut biased = [zero_i32(); L2_SIZE / CHUNK_SIZE_I32];
        for (i, biased) in biased.iter_mut().enumerate() {
            *biased = unsafe { load_i32(l1_bias.add(i * CHUNK_SIZE_I32)) };
            for unrolled in &l1_intermediate {
                *biased = add_i32(*biased, unrolled[i]);
            }
        }

        skip = to_array_i32(biased[0])[0];

        let hi = set1_i16(1 << (SHIFT + Q_BITS));

        for i in 0..L2_SIZE / CHUNK_SIZE_I16 {
            let packed = packus_i32(biased[i], biased[i + L2_SIZE / CHUNK_SIZE_I16]);
            let clamped = min_u16(packed, hi);
            let screlu = mulhi_i16(clamped, clamped);

            unsafe { store_i16(l1_out.0.as_mut_ptr().add(i * CHUNK_SIZE_I16), screlu) };
        }
    }

    let mut l2_intermediate = [[zero_i32(); 2]; L3_SIZE / CHUNK_SIZE_I32];
    {
        let l2_weights = NET.l2w[output_bucket].as_ptr();
        let l1_pairs = l1_out.0.as_ptr();

        for i in (0..L2_PAIR_COUNT).step_by(2) {
            let l1_vec_0 = unsafe { set1_i16x2(l1_pairs.add(i * 2)) };
            let l1_vec_1 = unsafe { set1_i16x2(l1_pairs.add((i + 1) * 2)) };
            for (j, intermediate) in l2_intermediate.iter_mut().enumerate() {
                let l2_weight_0 = unsafe { load_i16(l2_weights.add((i * L3_SIZE + j * CHUNK_SIZE_I32) * 2)) };
                let l2_weight_1 = unsafe { load_i16(l2_weights.add(((i + 1) * L3_SIZE + j * CHUNK_SIZE_I32) * 2)) };
                *intermediate = dpwssdx2_i32(*intermediate, l1_vec_0, l2_weight_0, l1_vec_1, l2_weight_1);
            }
        }
    }

    let l2_intermediate = l2_intermediate.map(reduce_dpwssd_i32);

    let mut l3_sum = zero_i32();
    {
        let l2_biases = NET.l2b[output_bucket].as_ptr();
        let l3_weights = NET.l3w[output_bucket].as_ptr();

        let skip_values = l1_out.0.as_ptr();

        let lo = zero_i32();
        let hi3 = set1_i32(1 << (3 * Q_BITS));

        for (i, &intermediate) in l2_intermediate.iter().enumerate() {
            let l2_bias = unsafe { load_i32(l2_biases.add(i * CHUNK_SIZE_I32)) };
            let l3_weight = unsafe { load_i32(l3_weights.add(i * CHUNK_SIZE_I32)) };

            let skip = unsafe { load_widen_i16_i32(skip_values.add(i * CHUNK_SIZE_I32)) };

            let biased = add_i32(intermediate, l2_bias);
            let activated = min_i32(max_i32(biased, lo), hi3);
            let skipped = add_i32(shift_left_i32::<Q_BITS>(skip), activated);

            l3_sum = add_i32(l3_sum, mul_i32(skipped, l3_weight));
        }
    }

    let bias = NET.l3b[output_bucket];
    let skip = i64::from(skip) * i64::from(Q * Q * Q * Q / (Q * 256));

    let out = i64::from(hsum_i32(l3_sum) + bias) + skip;
    let out = (out as f64 / f64::from(Q * Q * Q * Q)) as f32;

    1.0 / (1.0 + (-out).exp())
}

#[must_use]
fn accumulate(pos: &Position) -> Align64<[i16; L1_SIZE]> {
    let mut acc = Align64([0; L1_SIZE]);

    let mut rows = Updates::<L1_SIZE, UPDATE_CHUNK>::default();
    gather_rows(&mut rows, &NET.ft, pos);

    rows.apply(&mut acc.0, &NET.ft.b);

    acc
}

#[must_use]
fn calc_output_bucket(pos: &Position) -> usize {
    let w_reserves = pos.flats_in_hand(Color::White);
    let b_reserves = pos.flats_in_hand(Color::Black);
    let min_reserves = w_reserves.min(b_reserves);
    let bucket = [
        0, 0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    ][min_reserves as usize];
    let stm_komi = i32::from(pos.half_komi()) * pos.stm().sign();
    let komi_bucket = (-stm_komi - i32::from(Position::MIN_HALF_KOMI)) as usize;
    komi_bucket * RESERVE_BUCKET_COUNT + bucket
}
