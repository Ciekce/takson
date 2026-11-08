use crate::core::{Color, Piece, PieceType, Square};
use crate::eval::simd::{
    CHUNK_SIZE_I16, load_i16, max_i16, min_i16, mulhi_shift_i16, packus_i16, set1_i16, store_u8, zero_i16,
};
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
struct Align64<T>(T);

const Q0: i32 = 255;

const PACKUS_LANE: usize = 8;
const PACKUS_WIDTH: usize = 32;

#[must_use]
const fn packus_position(neuron: usize) -> usize {
    let chunk = neuron % (2 * PACKUS_WIDTH);
    let lane = chunk / (2 * PACKUS_LANE);
    let half = chunk % (2 * PACKUS_LANE) / PACKUS_LANE;
    neuron - chunk + half * PACKUS_WIDTH + lane * PACKUS_LANE + chunk % PACKUS_LANE
}

const EXPLICIT_MULHI_SHIFT_UP: i32 = 7;
const IMPLIED_MULHI_SHIFT_DOWN: i32 = 16;
const MULHI_SHIFT: i32 = EXPLICIT_MULHI_SHIFT_UP - IMPLIED_MULHI_SHIFT_DOWN;

fn activate_ft<const L1_SIZE: usize, const L1_PAIR_COUNT: usize>(dst: &mut [u8; L1_PAIR_COUNT], acc: &[i16; L1_SIZE]) {
    const {
        assert!(L1_SIZE == L1_PAIR_COUNT * 2);
    }

    let lo = zero_i16();
    let hi = set1_i16(Q0 as i16);

    let values = acc.as_ptr();
    let out = dst.as_mut_ptr();

    let activate = |offset: usize| {
        let a = unsafe { load_i16(values.add(offset)) };
        let b = unsafe { load_i16(values.add(offset + L1_PAIR_COUNT)) };

        let a = min_i16(max_i16(a, lo), hi);
        let b = min_i16(b, hi);

        mulhi_shift_i16::<EXPLICIT_MULHI_SHIFT_UP>(a, b)
    };

    for i in (0..L1_PAIR_COUNT).step_by(2 * CHUNK_SIZE_I16) {
        let offset_1 = packus_position(i);
        let offset_2 = packus_position(i + PACKUS_LANE);
        let p = packus_i16(activate(offset_1), activate(offset_2));
        unsafe { store_u8(out.add(i), p) };
    }

    debug_assert!(dst.iter().all(|&e| e <= i8::MAX as u8));
}

const MAX_UPDATE_ROWS: usize = 128;

#[derive(Default)]
struct Updates<const L1_SIZE: usize, const UPDATE_CHUNK: usize> {
    adds: ArrayVec<&'static [i16; L1_SIZE], MAX_UPDATE_ROWS>,
    subs: ArrayVec<&'static [i16; L1_SIZE], MAX_UPDATE_ROWS>,
}

impl<const L1_SIZE: usize, const UPDATE_CHUNK: usize> Updates<L1_SIZE, UPDATE_CHUNK> {
    fn apply(&self, dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
        let paired = self.adds.len().min(self.subs.len());

        for offset in (0..L1_SIZE).step_by(UPDATE_CHUNK) {
            let mut chunk: [i16; UPDATE_CHUNK] = src[offset..offset + UPDATE_CHUNK].try_into().unwrap();

            for (&add, &sub) in self.adds.iter().zip(self.subs.iter()) {
                for ((v, add), sub) in chunk.iter_mut().zip(&add[offset..]).zip(&sub[offset..]) {
                    *v += *add - *sub;
                }
            }

            for &add in &self.adds[paired..] {
                for (v, add) in chunk.iter_mut().zip(&add[offset..]) {
                    *v += *add;
                }
            }

            for &sub in &self.subs[paired..] {
                for (v, sub) in chunk.iter_mut().zip(&sub[offset..]) {
                    *v -= *sub;
                }
            }

            dst[offset..offset + UPDATE_CHUNK].copy_from_slice(&chunk);
        }
    }
}
