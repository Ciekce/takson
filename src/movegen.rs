use crate::core::{Direction, PieceType, Square};
use crate::hits::find_hits;
use crate::position::Position;
use crate::takmove::Move;

fn generate_starting_moves(dst: &mut Vec<Move>, pos: &Position) {
    for sq in !pos.occ() {
        dst.push(Move::placement(PieceType::Flat, sq));
    }
}

fn generate_placements(dst: &mut Vec<Move>, pos: &Position) {
    let flats = pos.flats_in_hand(pos.stm());
    let caps = pos.caps_in_hand(pos.stm());

    if flats == 0 && caps == 0 {
        return;
    }

    for sq in !pos.occ() {
        if caps > 0 {
            dst.push(Move::placement(PieceType::Capstone, sq));
        }

        if flats > 0 {
            dst.push(Move::placement(PieceType::Flat, sq));
            dst.push(Move::placement(PieceType::Wall, sq));
        }
    }
}

fn do_spreads(dst: &mut Vec<Move>, sq: Square, dir: Direction, lsb: u16, mut pattern: u16, dist: u32, limit: u16) {
    assert!(dist > 0);
    while pattern < limit {
        dst.push(Move::spread(sq, dir, pattern));
        if pattern.count_ones() == dist {
            pattern += pattern.isolate_lowest_one();
        } else {
            pattern += lsb;
        }
    }
}

fn generate_spreads(dst: &mut Vec<Move>, pos: &Position) {
    for sq in pos.color_bb(pos.stm()) {
        let top_pt = pos.top_pt(sq).unwrap();
        let max = pos.height(sq).min(6);

        let start_bit = (1 << Position::CARRY_LIMIT) >> max;

        let hits = find_hits(pos.all_blockers(), sq);

        for dir in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
            let (mut dist, hit_sq) = hits[dir];

            if dist == 0 {
                continue;
            }

            let mut limit = 1 << Position::CARRY_LIMIT;

            match pos.top_pt(hit_sq) {
                Some(PieceType::Wall) => {
                    if top_pt == PieceType::Capstone {
                        // Can smash - generate spreads here with msb set
                        do_spreads(
                            dst,
                            sq,
                            dir,
                            start_bit,
                            1 << (Position::CARRY_LIMIT - 1),
                            dist as u32,
                            limit,
                        );
                        limit >>= 1;
                    }
                    dist -= 1;
                }
                Some(PieceType::Capstone) => {
                    dist -= 1;
                }
                _ => {}
            }

            if dist == 0 {
                continue;
            }

            do_spreads(dst, sq, dir, start_bit, start_bit, dist as u32, limit);
        }
    }
}

pub fn generate_moves(dst: &mut Vec<Move>, pos: &Position) {
    dst.clear();

    if pos.ply() < 2 {
        generate_starting_moves(dst, pos);
        return;
    }

    generate_placements(dst, pos);
    generate_spreads(dst, pos);
}
