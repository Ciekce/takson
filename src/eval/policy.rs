use super::simd::*;
use crate::core::{PieceType, Square, Transform, normalizer};
use crate::eval::*;
use crate::position::Position;
use crate::takmove::Move;

const SPREAD_PATTERNS: usize = 62;
const MOVE_COUNT: usize = Square::COUNT * (PieceType::COUNT + 2 * SPREAD_PATTERNS);
const Q: i32 = 128;

const L1_SIZE: usize = 4096;

const L1_PAIR_COUNT: usize = L1_SIZE / 2;

const UPDATE_CHUNK: usize = 256;
const GROUP_SIZE: usize = 4;
const L1_UNROLL: usize = 4;

#[repr(C, align(64))]
struct Network {
    ft: FeatureTransformer<L1_SIZE, i8>,
    l1w: [[i8; L1_SIZE / 2]; MOVE_COUNT],
    l1b: [i32; MOVE_COUNT],
    scale: f32,
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
        let mut activated_ft = Align64([0; L1_SIZE / 2]);
        accumulate_activate(&mut activated_ft.0, pos);
        let transform = normalizer(pos.cap_sq(pos.stm()));
        Self {
            activations: activated_ft,
            transform,
            ply: pos.ply(),
        }
    }

    #[must_use]
    pub fn get(&self, mv: Move) -> f32 {
        self.compute_move_logits([mv])[0]
    }

    pub fn iter_policy(&self, moves: &[Move]) -> impl Iterator<Item = f32> {
        let (groups, rest) = moves.as_chunks::<GROUP_SIZE>();
        groups
            .iter()
            .flat_map(move |&group| self.compute_move_logits(group))
            .chain(rest.iter().map(move |&mv| self.get(mv)))
    }

    #[inline(always)]
    fn compute_move_logits<const N: usize>(&self, moves: [Move; N]) -> [f32; N] {
        let mut idx = [0; N];
        let mut weights = [std::ptr::null::<i8>(); N];
        for j in 0..N {
            idx[j] = self.move_idx(moves[j]);
            weights[j] = NET.l1w[idx[j]].as_ptr();
        }
        let activations = self.activations.0.as_ptr();
        let mut l1_intermediate = [[zero_i32(); L1_UNROLL]; N];
        let mut i_outer = 0;
        while i_outer < L1_PAIR_COUNT {
            for i_inner in 0..L1_UNROLL {
                let offs = i_outer + i_inner * CHUNK_SIZE_I8;
                let a = unsafe { load_u8(activations.add(offs)) };
                for j in 0..N {
                    let w = unsafe { load_i8(weights[j].add(offs)) };
                    l1_intermediate[j][i_inner] = dpbusd_i32(l1_intermediate[j][i_inner], a, w);
                }
            }
            i_outer += L1_UNROLL * CHUNK_SIZE_I8;
        }
        std::array::from_fn(|j| {
            let sum = hsum_i32(l1_intermediate[j].into_iter().reduce(add_i32).unwrap());
            let out = (sum + NET.l1b[idx[j]]) as f32 * NET.scale;
            match self.ply < 2 {
                true => -out,
                false => out,
            }
        })
    }

    #[inline(always)]
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

fn accumulate_activate(dst: &mut [u8; L1_PAIR_COUNT], pos: &Position) {
    let mut rows = Updates::<L1_SIZE, UPDATE_CHUNK, i8>::default();
    gather_rows(&mut rows, &NET.ft, pos);
    for offset in (0..L1_PAIR_COUNT).step_by(UPDATE_CHUNK) {
        let mut chunk_0 = Align64([0; UPDATE_CHUNK]);
        let mut chunk_1 = Align64([0; UPDATE_CHUNK]);
        for j in 0..UPDATE_CHUNK {
            chunk_0.0[j] = i16::from(NET.ft.b[offset + j]);
            chunk_1.0[j] = i16::from(NET.ft.b[L1_PAIR_COUNT + offset + j]);
        }
        for row in &rows.adds {
            for j in 0..UPDATE_CHUNK {
                chunk_0.0[j] += i16::from(row[offset + j]);
                chunk_1.0[j] += i16::from(row[L1_PAIR_COUNT + offset + j]);
            }
        }
        let activate = |offset: usize| {
            let a = unsafe { load_i16(chunk_0.0.as_ptr().add(offset)) };
            let b = unsafe { load_i16(chunk_1.0.as_ptr().add(offset)) };
            let a = min_i16(max_i16(a, zero_i16()), set1_i16(Q as i16));
            let b = min_i16(b, set1_i16(Q as i16 - 1));
            mulhi_shift_i16::<8>(a, adds_i16(b, b))
        };
        for i in (0..UPDATE_CHUNK).step_by(2 * CHUNK_SIZE_I16) {
            let packed = packus_i16(activate(packus_position(i)), activate(packus_position(i + PACKUS_LANE)));
            unsafe { store_u8(dst.as_mut_ptr().add(offset + i), packed) };
        }
    }
}
