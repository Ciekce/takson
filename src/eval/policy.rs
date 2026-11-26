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

const PICK_PATTERNS: usize = 567;
const LAND_PATTERNS: usize = 2 * 363;

#[repr(C, align(64))]
struct Network {
    ft: FeatureTransformer<L1_SIZE, i8>,
    l1w: [[i8; L1_SIZE / 2]; MOVE_COUNT],
    pick: [[i8; L1_PAIR_COUNT]; PICK_PATTERNS],
    land: [[i8; L1_PAIR_COUNT]; LAND_PATTERNS],
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
    pub fn get(&self, pos: &Position, mv: Move) -> f32 {
        self.compute_move_logits(pos, [mv])[0]
    }

    pub fn iter_policy(&self, pos: &Position, moves: &[Move]) -> impl Iterator<Item = f32> {
        let (groups, rest) = moves.as_chunks::<GROUP_SIZE>();
        groups
            .iter()
            .flat_map(|&group| self.compute_move_logits(pos, group))
            .chain(rest.iter().map(|&mv| self.get(pos, mv)))
    }

    #[inline(always)]
    fn compute_move_logits<const N: usize>(&self, pos: &Position, moves: [Move; N]) -> [f32; N] {
        const ZERO_ROW: Align64<[i8; L1_PAIR_COUNT]> = Align64([0; L1_PAIR_COUNT]);

        let mut idx = [0; N];
        let mut weights = [std::ptr::null::<i8>(); N];
        let mut picks = [ZERO_ROW.0.as_ptr(); N];
        let mut lands = [ZERO_ROW.0.as_ptr(); N];
        let mut any_spread = false;
        for j in 0..N {
            idx[j] = self.move_idx(moves[j]);
            weights[j] = NET.l1w[idx[j]].as_ptr();
            if moves[j].is_spread() {
                let (pick, land) = pick_land_indices(pos, moves[j]);
                picks[j] = NET.pick[pick].as_ptr();
                lands[j] = NET.land[land].as_ptr();
                any_spread = true;
            }
        }
        let sums = match any_spread {
            true => self.dot::<N, true>(weights, picks, lands),
            false => self.dot::<N, false>(weights, picks, lands),
        };
        std::array::from_fn(|j| {
            let out = (sums[j] + NET.l1b[idx[j]]) as f32 * NET.scale;
            match self.ply < 2 {
                true => -out,
                false => out,
            }
        })
    }

    #[inline]
    fn dot<const N: usize, const PATTERNS: bool>(
        &self,
        weights: [*const i8; N],
        picks: [*const i8; N],
        lands: [*const i8; N],
    ) -> [i32; N] {
        let activations = self.activations.0.as_ptr();
        let mut l1_intermediate = [[zero_i32(); L1_UNROLL]; N];
        let mut i_outer = 0;
        while i_outer < L1_PAIR_COUNT {
            for i_inner in 0..L1_UNROLL {
                let offs = i_outer + i_inner * CHUNK_SIZE_I8;
                let a = unsafe { load_u8(activations.add(offs)) };
                for j in 0..N {
                    let mut w = unsafe { load_i8(weights[j].add(offs)) };
                    if PATTERNS {
                        let (pick, land) = unsafe { (load_i8(picks[j].add(offs)), load_i8(lands[j].add(offs))) };
                        w = adds_i8(w, adds_i8(pick, land));
                    }
                    l1_intermediate[j][i_inner] = dpbusd_i32(l1_intermediate[j][i_inner], a, w);
                }
            }
            i_outer += L1_UNROLL * CHUNK_SIZE_I8;
        }
        l1_intermediate.map(|intermediate| hsum_i32(intermediate.into_iter().reduce(add_i32).unwrap()))
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

#[inline(always)]
fn pick_land_indices(pos: &Position, mv: Move) -> (usize, usize) {
    let src = mv.sq();
    let stm = pos.stm();
    let taken = Position::CARRY_LIMIT - mv.pattern().trailing_zeros();
    let colors = pos.colors_for(src, stm) as usize;
    let mask = (1 << taken - 1) - 1;
    let pick = mask + ((colors >> 1) & mask);
    let left = if taken == u32::from(pos.height(src)) {
        0
    } else {
        2 - ((colors >> taken) & 1)
    };
    let pick = (pick * 3 + pos.top_pt(src).unwrap().idx()) * 3 + left;

    let dst = mv.spread_dest();
    let mut sq = dst;
    let mut land = 0;
    for _ in 0..mv.spread_length() {
        let digit = match pos.top_color(sq) {
            None => 1,
            Some(color) if color == stm => 2,
            Some(_) => 3,
        };
        land = 3 * land + digit;
        sq = sq.shift(mv.dir().flip()).unwrap();
    }
    land -= 1;

    (pick, 2 * land + usize::from(pos.top_pt(dst) == Some(PieceType::Wall)))
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
