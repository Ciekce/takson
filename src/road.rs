cfg_select! {
    target_feature = "avx512f" => {
        mod avx512;
        use avx512::influence as influence_impl;
    }
    target_feature = "avx2" => {
        mod avx2;
        use avx2::influence as influence_impl;
    }
    target_feature = "sse4.2" => {
        mod sse;
        use sse::influence as influence_impl;
    }
    _ => {
        mod scalar;
        use scalar::influence as influence_impl;
    }
}

use crate::bitboard::Bitboard;

#[must_use]
pub fn influence(road_occ: Bitboard) -> (bool, [Bitboard; 4]) {
    let mut edges = Bitboard::EDGES.map(|edge| road_occ & edge);

    edges[0] |= edges[0] >> 6 & road_occ;
    edges[1] |= edges[1] << 6 & road_occ;
    edges[2] |= edges[2] << 1 & road_occ;
    edges[3] |= edges[3] >> 1 & road_occ;

    let result = influence_impl(road_occ, &mut edges);

    (result, edges)
}
