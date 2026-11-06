use crate::limit::Limits;
use crate::movegen::generate_moves;
use crate::position::{FlatCountOutcome, Position};
use crate::takmove::Move;
use crate::tei::{TeiOptions, score_to_cp};
use crate::value::eval_value;
use std::collections::HashSet;
use std::time::Instant;

pub const MAX_DEPTH: i32 = 1;

pub struct Searcher {
    //
}

impl Searcher {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_search(
        &mut self,
        pos: &Position,
        new_key_history: &[u64],
        start_time: Instant,
        _limits: Limits,
        _max_depth: i32,
        moves_to_search: &[Move],
        moves_to_exclude: &HashSet<Move>,
        _options: &TeiOptions,
    ) {
        let mut key_history = new_key_history.to_vec();
        key_history.push(pos.key());

        let mut moves = Vec::new();
        get_root_moves(&mut moves, pos, moves_to_search, moves_to_exclude);

        let mut nodes = 1usize;

        let mut best_move = None;
        let mut best_score = -1.0;

        for mv in moves {
            nodes += 1;

            let mut new_pos = *pos;
            new_pos.apply_move(mv);

            let score = if let Some(state) = check_terminal_state(&key_history, &new_pos, mv) {
                match state {
                    TerminalState::Win => 1.0,
                    TerminalState::Draw => 0.5,
                    TerminalState::Loss => 0.0,
                }
            } else {
                1.0 - eval_value(&new_pos)
            };

            if score > best_score {
                best_move = Some(mv);
                best_score = score;
            }
        }

        let best_move = best_move.unwrap();

        let time = start_time.elapsed().as_secs_f64();
        let [w, d, l] =
            [best_score, 0.0, 1.0 - best_score].map(|score| (score * 1000.0).round() as i32);
        let cp = score_to_cp(best_score);

        println!(
            "info depth 1 seldepth 1 time {} nodes {} nps {} score cp {} wdl {} {} {} pv {}",
            (time * 1000.0) as usize,
            nodes,
            (nodes as f64 / time) as usize,
            cp,
            w,
            d,
            l,
            best_move
        );
        println!("bestmove {}", best_move);
    }
}

fn get_root_moves(
    moves: &mut Vec<Move>,
    root_pos: &Position,
    moves_to_search: &[Move],
    moves_to_exclude: &HashSet<Move>,
) {
    if !moves_to_search.is_empty() {
        print!("info string searchmoves:");
        for &mv in moves_to_search {
            assert!(root_pos.is_legal(mv));
            print!(" {}", mv);
        }
        println!();

        moves.extend_from_slice(moves_to_search);
    } else {
        generate_moves(moves, root_pos);
    }

    if !moves_to_exclude.is_empty() {
        print!("info string excludemoves:");
        moves.retain(|mv| {
            let keep = !moves_to_exclude.contains(mv);
            if !keep {
                print!(" {}", *mv);
            }
            keep
        });
        println!();
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum TerminalState {
    Win,
    Draw,
    Loss,
}

fn check_terminal_state(
    key_history: &[u64],
    pos: &Position,
    prev_move: Move,
) -> Option<TerminalState> {
    // player we care about: the one that made the move
    let stm = pos.stm().flip();

    if pos.has_road(stm) {
        return Some(TerminalState::Win);
    }

    if pos.has_road(stm.flip()) {
        return Some(TerminalState::Loss);
    }

    match pos.count_flats() {
        FlatCountOutcome::None => {}
        FlatCountOutcome::Draw => return Some(TerminalState::Draw),
        FlatCountOutcome::Win(color) => {
            return if color == stm {
                Some(TerminalState::Win)
            } else {
                Some(TerminalState::Loss)
            };
        }
    }

    if prev_move.is_spread() {
        let mut repetitions = 0;
        for &key in key_history.iter().rev() {
            if key == pos.key() {
                repetitions += 1;
                if repetitions == 2 {
                    return Some(TerminalState::Draw);
                }
            }
        }
    }

    None
}
