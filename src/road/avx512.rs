use crate::bitboard::Bitboard;
use std::arch::x86_64::*;

#[must_use]
pub(super) fn influence(road_occ: Bitboard, edges: &mut [Bitboard; 4]) -> bool {
    unsafe { influence_avx512(road_occ, edges) }
}

#[must_use]
#[target_feature(enable = "avx512f")]
fn influence_avx512(road_occ: Bitboard, edges: &mut [Bitboard; 4]) -> bool {
    let mut masks = _mm512_broadcast_i64x4(unsafe { std::mem::transmute::<[Bitboard; 4], __m256i>(*edges) });

    let left_edge = Bitboard::LEFT_EDGE.raw() as i64;
    let right_edge = Bitboard::RIGHT_EDGE.raw() as i64;

    let shifts = _mm512_set_epi64(1, 1, 1, 1, 6, 6, 6, 6);
    let left_edges = _mm512_set_epi64(left_edge, left_edge, left_edge, left_edge, 0, 0, 0, 0);
    let right_edges = _mm512_set_epi64(right_edge, right_edge, right_edge, right_edge, 0, 0, 0, 0);

    let road_occ = _mm512_set1_epi64(road_occ.raw() as i64);

    let calc_next_masks = |masks| {
        let next_masks_ul = _mm512_andnot_si512(left_edges, _mm512_sllv_epi64(masks, shifts));
        let next_masks_dr = _mm512_andnot_si512(right_edges, _mm512_srlv_epi64(masks, shifts));
        let next_masks = _mm512_or_si512(masks, _mm512_or_si512(next_masks_ul, next_masks_dr));

        _mm512_or_si512(
            next_masks,
            _mm512_shuffle_i64x2::<_MM_PERM_BADC>(next_masks, next_masks),
        )
    };

    let unchanged = |masks, next_masks| {
        let grown = _mm512_andnot_si512(masks, next_masks);
        _mm512_test_epi64_mask(grown, grown) == 0
    };

    let mut influence_masks = calc_next_masks(masks);
    let next_masks = _mm512_and_si512(influence_masks, road_occ);

    let result = 'exit: {
        if unchanged(masks, next_masks) {
            break 'exit false;
        }

        masks = next_masks;

        loop {
            influence_masks = calc_next_masks(masks);
            let next_masks = _mm512_and_si512(influence_masks, road_occ);
            let swizzled = _mm512_shuffle_epi32::<_MM_PERM_BADC>(next_masks);

            if _mm512_test_epi64_mask(next_masks, swizzled) != 0 {
                break 'exit true;
            }

            if unchanged(masks, next_masks) {
                break 'exit false;
            }

            masks = next_masks;
        }
    };

    let influence_masks = _mm256_or_si256(_mm512_castsi512_si256(influence_masks), unsafe {
        std::mem::transmute::<[Bitboard; 4], __m256i>(Bitboard::EDGES)
    });

    *edges = unsafe { std::mem::transmute::<__m256i, [u64; 4]>(influence_masks) }.map(Bitboard::from_raw);

    result
}
