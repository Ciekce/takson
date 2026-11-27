use crate::core::{Color, Piece, PieceType, Square, normalizer};
use crate::eval::simd::{
    CHUNK_SIZE_I16, adds_i16, load_i16, max_i16, min_i16, mulhi_shift_i16, packus_i16, set1_i16, store_u8, zero_i16,
};
use crate::position::Position;
use arrayvec::ArrayVec;

pub mod policy;
pub mod value;

cfg_select! {
    all(target_feature = "avx512f", target_feature = "avx512bw") => {
        #[path = "simd/avx512.rs"]
        mod simd;
    }
    target_feature = "avx2" => {
        #[path = "simd/avx2.rs"]
        mod simd;
    }
    target_feature = "sse4.1" => {
        #[path = "simd/sse41.rs"]
        mod simd;
    }
    all(target_arch = "aarch64", target_feature = "neon") => {
        #[path = "simd/neon.rs"]
        mod simd;
    }
    all(target_arch = "wasm32", target_feature = "simd128") => {
        #[path = "simd/wasm.rs"]
        mod simd;
    }
    _ => {
        #[path = "simd/scalar.rs"]
        mod simd;
    }
}

const TOP_INPUT_SIZE: usize = Color::COUNT * PieceType::COUNT * Square::COUNT;
const STACK_INPUT_SIZE: usize = 12 * Color::COUNT * PieceType::COUNT * Square::COUNT * Color::COUNT;
const RESERVES_INPUT_SIZE: usize = 30 * Color::COUNT;

#[must_use]
fn top_feature_idx(perspective: Color, side: Color, pt: PieceType, sq: Square) -> usize {
    (usize::from(side != perspective) * PieceType::COUNT + pt.idx()) * Square::COUNT + sq.idx()
}

#[must_use]
fn stack_feature_idx(perspective: Color, top: Piece, side: Color, depth: u8, sq: Square) -> usize {
    debug_assert!(depth < 12);
    let top_idx = usize::from(top.color() != perspective) * PieceType::COUNT + top.piece_type().idx();
    ((top_idx * Color::COUNT + usize::from(side != perspective)) * Square::COUNT + sq.idx()) * 12 + depth as usize
}

#[must_use]
fn reserves_feature_idx(perspective: Color, side: Color, reserves: u8) -> usize {
    usize::from(side != perspective) * 30 + (reserves as usize - 1)
}

#[repr(C, align(64))]
struct FeatureTransformer<const L1_SIZE: usize, T = i16> {
    w_top: [[T; L1_SIZE]; TOP_INPUT_SIZE],
    w_stack: [[T; L1_SIZE]; STACK_INPUT_SIZE],
    w_reserves: [[T; L1_SIZE]; RESERVES_INPUT_SIZE],
    b: [T; L1_SIZE],
}

#[repr(C, align(64))]
struct Align64<T>(T);

const PACKUS_LANE: usize = 8;
const PACKUS_WIDTH: usize = 32;

#[must_use]
const fn packus_position(neuron: usize) -> usize {
    let chunk = neuron % (2 * PACKUS_WIDTH);
    let lane = chunk / (2 * PACKUS_LANE);
    let half = chunk % (2 * PACKUS_LANE) / PACKUS_LANE;
    neuron - chunk + half * PACKUS_WIDTH + lane * PACKUS_LANE + chunk % PACKUS_LANE
}

const IMPLIED_MULHI_SHIFT_DOWN: i32 = 16;

const MAX_UPDATE_ROWS: usize = 128;

impl<const L1_SIZE: usize, T: Copy> FeatureTransformer<L1_SIZE, T>
where
    i16: From<T>,
{
    #[inline]
    fn accumulate_activate<const EXPLICIT_MULHI_SHIFT_UP: i32, const A_MAX: i16, const B_MAX: i16>(
        &self,
        dst: &mut [u8],
        pos: &Position,
    ) {
        const UPDATE_CHUNK: usize = cfg_select! {
            target_feature = "avx512f" => 512,
            _ => 256,
        };

        assert_eq!(dst.len(), L1_SIZE / 2);

        let mut rows = ArrayVec::<_, MAX_UPDATE_ROWS>::new();
        gather_rows(&mut rows, self, pos);

        let lo = zero_i16();
        let a_max = set1_i16(A_MAX);
        let b_max = set1_i16(B_MAX);

        for offset in (0..L1_SIZE / 2).step_by(UPDATE_CHUNK) {
            let mut chunk_0 = Align64([0; UPDATE_CHUNK]);
            let mut chunk_1 = Align64([0; UPDATE_CHUNK]);
            for (v, &bias) in chunk_0.0.iter_mut().zip(&self.b[offset..]) {
                *v = i16::from(bias);
            }
            for (v, &bias) in chunk_1.0.iter_mut().zip(&self.b[L1_SIZE / 2 + offset..]) {
                *v = i16::from(bias);
            }
            for row in &rows {
                for (v, &weight) in chunk_0.0.iter_mut().zip(&row[offset..]) {
                    *v += i16::from(weight);
                }
                for (v, &weight) in chunk_1.0.iter_mut().zip(&row[L1_SIZE / 2 + offset..]) {
                    *v += i16::from(weight);
                }
            }

            let activate = |offset: usize| {
                let a = unsafe { load_i16(chunk_0.0.as_ptr().add(offset)) };
                let b = unsafe { load_i16(chunk_1.0.as_ptr().add(offset)) };
                let a = min_i16(max_i16(a, lo), a_max);
                let b = min_i16(b, b_max);
                mulhi_shift_i16::<EXPLICIT_MULHI_SHIFT_UP>(a, adds_i16(b, b))
            };

            for i in (0..UPDATE_CHUNK).step_by(2 * CHUNK_SIZE_I16) {
                let packed = packus_i16(activate(packus_position(i)), activate(packus_position(i + PACKUS_LANE)));
                unsafe { store_u8(dst.as_mut_ptr().add(offset + i), packed) };
            }
        }

        debug_assert!(dst.iter().all(|&e| e <= i8::MAX as u8));
    }
}

fn gather_rows<'a, const L1_SIZE: usize, T>(
    rows: &mut ArrayVec<&'a [T; L1_SIZE], MAX_UPDATE_ROWS>,
    ft: &'a FeatureTransformer<L1_SIZE, T>,
    pos: &Position,
) {
    let perspective = pos.stm();

    let transform = normalizer(pos.cap_sq(perspective));

    for side in [Color::White, Color::Black] {
        for sq in pos.color_bb(side) {
            let perspective_sq = sq.transform(transform);
            let pt = pos.top_pt(sq).unwrap();
            rows.push(&ft.w_top[top_feature_idx(perspective, side, pt, perspective_sq)]);
        }
    }

    for sq in pos.occ() {
        let height = pos.height(sq);

        if height == 1 {
            continue;
        }

        let top = pos.top(sq).unwrap();
        let perspective_sq = sq.transform(transform);

        let mut colors = pos.colors(sq) >> 1;
        let count = height.min(13) - 1;

        for depth in 0..count {
            let color = Color::from_raw((colors & 0x1) as u8).unwrap();
            rows.push(&ft.w_stack[stack_feature_idx(perspective, top, color, depth, perspective_sq)]);
            colors >>= 1;
        }
    }

    for color in [Color::White, Color::Black] {
        let reserves = pos.flats_in_hand(color);

        if reserves == 0 {
            continue;
        }

        rows.push(&ft.w_reserves[reserves_feature_idx(perspective, color, reserves)]);
    }
}
