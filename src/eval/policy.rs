use super::simd::*;
use crate::core::{Color, PieceType, Square, Transform, normalizer};
use crate::eval::*;
use crate::position::Position;
use crate::takmove::Move;

const SPREAD_PATTERNS: usize = 62;
const MOVE_COUNT: usize = Square::COUNT * (PieceType::COUNT + 2 * SPREAD_PATTERNS);
const DIVISOR: f32 = 2048.0;

const L1_SIZE: usize = 2560;

const L1_PAIR_COUNT: usize = L1_SIZE / 2;

#[cfg(target_feature = "avx512f")]
const UPDATE_CHUNK: usize = L1_SIZE / 4;
#[cfg(not(target_feature = "avx512f"))]
const UPDATE_CHUNK: usize = 256;

#[repr(C, align(64))]
struct Network {
    ft: FeatureTransformer<L1_SIZE>,
    l1w: [[i8; L1_SIZE / 2]; MOVE_COUNT],
    l1b: [i32; MOVE_COUNT],
}

static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("POLICYFILE"))) };

pub struct PolicyContext {
    activations: Align64<[u8; L1_SIZE / 2]>,
    transform: Transform,
    ply: u16,
}

impl PolicyContext {
    #[must_use]
    pub fn new(pos: &Position) -> Self {
        let acc = accumulate(pos);
        let mut activated_ft = Align64([0; L1_SIZE / 2]);
        activate_ft::<L1_SIZE, L1_PAIR_COUNT>(&mut activated_ft.0, &acc.0);
        let transform = normalizer(pos.cap_sq(pos.stm()));
        Self {
            activations: activated_ft,
            transform,
            ply: pos.ply(),
        }
    }

    #[must_use]
    pub fn get(&self, mv: Move) -> f32 {
        let idx = self.move_idx(mv);
        let weights = NET.l1w[idx].iter().zip(&self.activations.0);
        let sum: i32 = weights.map(|(&w, &a)| i32::from(w) * i32::from(a)).sum();
        let out = sum + NET.l1b[idx];
        match self.ply < 2 {
            true => -out as f32 / DIVISOR,
            false => out as f32 / DIVISOR,
        }
    }

    fn move_idx(&self, mv: Move) -> usize {
        let src = mv.sq().transform(self.transform);

        if !mv.is_spread() {
            return src.idx() * PieceType::COUNT + mv.pt().idx();
        }

        let dst = mv.spread_dest().transform(self.transform);
        let horizontal = usize::from(src.rank() == dst.rank());

        let mut pattern = mv.pattern() as usize;
        if dst.idx() < src.idx() {
            pattern ^= 63;
        }

        Square::COUNT * PieceType::COUNT + (2 * src.idx() + horizontal) * SPREAD_PATTERNS + pattern - 1
    }
}

#[must_use]
fn accumulate(pos: &Position) -> Align64<[i16; L1_SIZE]> {
    let mut acc = Align64([0; L1_SIZE]);

    let mut rows = Updates::<L1_SIZE, UPDATE_CHUNK>::default();
    gather_rows(&mut rows, &NET.ft, pos);

    rows.apply(&mut acc.0, &NET.ft.b);

    acc
}
