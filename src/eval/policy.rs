use super::simd::*;
use crate::core::{Color, PieceType, Square, Transform, normalizer};
use crate::eval::*;
use crate::position::Position;
use crate::takmove::Move;

const SPREAD_PATTERNS: usize = 62;
const MOVE_COUNT: usize = Square::COUNT * (PieceType::COUNT + 2 * SPREAD_PATTERNS);
const DIVISOR: f32 = 2048.0;

const L1_SIZE: usize = 1280;

const L1_PAIR_COUNT: usize = L1_SIZE / 2;

#[cfg(target_feature = "avx512f")]
const UPDATE_CHUNK: usize = L1_SIZE / 2;
#[cfg(not(target_feature = "avx512f"))]
const UPDATE_CHUNK: usize = 256;

#[repr(C, align(64))]
struct Network {
    ftw_top: [[i16; L1_SIZE]; TOP_INPUT_SIZE],
    ftw_stack: [[i16; L1_SIZE]; STACK_INPUT_SIZE],
    ftw_reserves: [[i16; L1_SIZE]; RESERVES_INPUT_SIZE],
    ftb: [i16; L1_SIZE],
    l1w: [[i8; L1_SIZE]; MOVE_COUNT],
    l1b: [i32; MOVE_COUNT],
}

static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("POLICYFILE"))) };

pub struct PolicyContext {
    activations: Align64<[u8; L1_SIZE]>,
    transform: Transform,
    ply: u16,
}

impl PolicyContext {
    #[must_use]
    pub fn new(pos: &Position) -> Self {
        let acc = accumulate(pos);
        let mut activated_ft = Align64([0; L1_SIZE]);
        for (dst, acc) in activated_ft
            .0
            .as_chunks_mut::<L1_PAIR_COUNT>()
            .0
            .iter_mut()
            .zip(acc.0.as_chunks::<L1_SIZE>().0)
        {
            activate_ft::<L1_SIZE, L1_PAIR_COUNT>(dst, acc);
        }
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
fn accumulate(pos: &Position) -> Align64<[i16; L1_SIZE * 2]> {
    let mut acc = Align64([0; L1_SIZE * 2]);

    for (&perspective, values) in [pos.stm(), pos.stm().flip()]
        .iter()
        .zip(acc.0.as_chunks_mut::<L1_SIZE>().0)
    {
        let transform = normalizer(pos.cap_sq(perspective));

        let mut rows = Updates::<L1_SIZE, UPDATE_CHUNK>::default();

        for side in [Color::White, Color::Black] {
            for sq in pos.color_bb(side) {
                let perspective_sq = sq.transform(transform);
                let pt = pos.top_pt(sq).unwrap();
                rows.adds
                    .push(&NET.ftw_top[top_feature_idx(perspective, side, pt, perspective_sq)]);
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
                rows.adds
                    .push(&NET.ftw_stack[stack_feature_idx(perspective, top, color, depth, perspective_sq)]);
                colors >>= 1;
            }
        }

        for color in [Color::White, Color::Black] {
            let reserves = pos.flats_in_hand(color);

            if reserves == 0 {
                continue;
            }

            rows.adds
                .push(&NET.ftw_reserves[reserves_feature_idx(perspective, color, reserves)]);
        }

        rows.apply(values, &NET.ftb);
    }

    acc
}
