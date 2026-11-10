use crate::movegen::{MoveList, generate_moves};
use crate::position::Position;
use std::time::Instant;

fn do_perft(pos: &Position, depth: i32, movelists: &mut [MoveList]) -> usize {
    if depth <= 0 {
        return 1;
    }

    let (moves, movelists) = movelists.split_first_mut().unwrap();
    generate_moves(moves, pos);

    if depth == 1 {
        return moves.len();
    }

    let mut total = 0;

    for &mut mv in moves {
        debug_assert!(pos.is_legal(mv));

        let mut new_pos = *pos;
        new_pos.apply_move(mv);
        total += do_perft(&new_pos, depth - 1, movelists);
    }

    total
}

#[must_use]
pub fn perft(pos: &Position, depth: i32) -> usize {
    let mut movelists = vec![MoveList::new(); depth as usize];
    do_perft(pos, depth.max(1), &mut movelists)
}

pub fn split_perft(pos: &Position, depth: i32) {
    let depth = depth.max(1);

    let mut movelists = vec![MoveList::new(); depth as usize];

    let start = Instant::now();

    let (moves, movelists) = movelists.split_first_mut().unwrap();
    generate_moves(moves, pos);

    let mut total = 0;

    for &mut mv in moves {
        debug_assert!(pos.is_legal(mv));

        print!("{:9}  ", mv.to_string());

        let mut new_pos = *pos;
        new_pos.apply_move(mv);
        let value = do_perft(&new_pos, depth - 1, movelists);

        total += value;
        println!("{}", value);
    }

    let nps = (total as f64 / start.elapsed().as_secs_f64()) as usize;

    println!();
    println!("total: {}", total);
    println!("{} nps", nps);
}
