use crate::bitboard::Bitboard;
use crate::core::{Direction, Square};
#[cfg(target_feature = "bmi2")]
use std::arch::x86_64::_pdep_u64;

pub(super) const fn generate_mask(sq: Square) -> u64 {
    let mut mask = Bitboard::empty();

    let mut dir_idx = 0;
    while let Some(dir) = Direction::from_raw(dir_idx) {
        let mut dir_bb = Bitboard::empty();

        let mut sq = sq;

        while let Some(shifted) = sq.shift_checked(dir) {
            dir_bb.set_sq(shifted);
            sq = shifted;
        }

        mask = mask.or(dir_bb.and(Bitboard::edge(dir).cmpl()));

        dir_idx += 1;
    }

    mask.raw()
}

#[allow(unreachable_code)]
pub(super) fn pdep(v: u64, mask: u64) -> u64 {
    #[cfg(target_feature = "bmi2")]
    {
        return unsafe { _pdep_u64(v, mask) };
    }

    let mut mask = mask;

    let mut x = 0;
    let mut bit = 1;

    while mask != 0 {
        if (v & bit) != 0 {
            x |= mask.isolate_lowest_one();
        }

        bit <<= 1;
        mask &= mask - 1;
    }

    x
}
