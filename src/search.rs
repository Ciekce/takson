use crate::limit::Limits;
use crate::movegen::generate_moves;
use crate::position::{FlatCountOutcome, Position};
use crate::takmove::Move;
use crate::tei::{TeiOptions, score_to_cp};
use crate::value::eval_value;
use arrayvec::ArrayVec;
use std::collections::HashSet;
use std::ops::Range;
use std::time::Instant;

pub const MAX_DEPTH: i32 = 255;

pub const DEFAULT_TREE_SIZE_MIB: usize = 16;
pub const MAX_TREE_SIZE_MIB: usize = 131072;

#[derive(Copy, Clone, Default, Debug)]
struct Node {
    idx: u32,
    parent: u32,
    mv: Option<Move>,
    visits: usize,
    total_score: f64,
    first_child_idx: u32,
    child_count: u16,
    result: Option<TerminalState>,
}

impl Node {
    #[must_use]
    fn new(idx: u32, parent: u32, mv: Move) -> Self {
        Self {
            idx,
            parent,
            mv: Some(mv),
            visits: 0,
            total_score: 0.0,
            first_child_idx: 0,
            child_count: 0,
            result: None,
        }
    }

    #[must_use]
    fn q(&self) -> f32 {
        assert!(self.visits > 0);
        (self.total_score / self.visits as f64) as f32
    }

    #[must_use]
    fn children(&self) -> Range<usize> {
        assert!(self.child_count > 0);
        let start = self.first_child_idx as usize;
        let end = start + self.child_count as usize;
        start..end
    }

    #[must_use]
    fn is_terminal(&self) -> bool {
        self.result.is_some()
    }
}

pub struct Searcher {
    tree: Vec<Node>,
    pos: Position,
    key_history: Vec<u64>,
}

impl Searcher {
    #[must_use]
    pub fn new(tree_size_mib: usize) -> Self {
        let node_count = tree_size_mib * 1024 * 1024 / size_of::<Node>();
        Self {
            tree: Vec::with_capacity(node_count),
            pos: Position::startpos(),
            key_history: Vec::with_capacity(1024),
        }
    }

    pub fn set_tree_size(&mut self, size_mib: usize) {
        self.tree.clear();
        self.tree.shrink_to_fit();

        let node_count = size_mib * 1024 * 1024 / size_of::<Node>();
        self.tree = Vec::with_capacity(node_count);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_search(
        &mut self,
        root_pos: &Position,
        new_key_history: &[u64],
        start_time: Instant,
        limits: Limits,
        _max_depth: i32,
        moves_to_search: &[Move],
        moves_to_exclude: &HashSet<Move>,
        _options: &TeiOptions,
    ) {
        const REPORT_INTERVAL: usize = 65536;

        self.tree.clear();
        self.construct_root(root_pos, moves_to_search, moves_to_exclude);

        self.key_history.clear();
        self.key_history
            .reserve(new_key_history.len() + MAX_DEPTH as usize);
        self.key_history.extend_from_slice(new_key_history);

        let mut iters = 0usize;

        loop {
            self.pos = *root_pos;
            self.key_history.truncate(new_key_history.len());

            let Some(leaf) = self.select_expand() else {
                println!("info string Tree full, stopping");
                break;
            };

            assert_ne!(leaf, 0);

            let u = self.simulate(leaf);
            self.backprop(leaf, u);

            iters += 1;

            if iters.is_multiple_of(REPORT_INTERVAL) {
                self.report(start_time, iters);
            }

            if limits.should_stop(iters) {
                break;
            }
        }

        self.final_report(start_time, iters);
    }

    fn final_report(&self, start_time: Instant, iters: usize) {
        let first_best_child = self.report(start_time, iters);
        assert_ne!(first_best_child.idx, 0);

        println!();
        let root = &self.tree[0];
        println!("info string root V: {} Q: {}", root.visits, root.q());
        let mut total_visits = 0;
        for child_idx in self.tree[0].children() {
            let node = &self.tree[child_idx];
            println!(
                "info string {} V: {}, Q: {}",
                node.mv.unwrap(),
                node.visits,
                1.0 - node.q()
            );
            total_visits += node.visits;
        }
        assert_eq!(total_visits, root.visits);
        println!();

        println!("bestmove {}", first_best_child.mv.unwrap());
    }

    fn report(&self, start_time: Instant, iters: usize) -> &Node {
        let mut pv = ArrayVec::new();
        let first_best_child = self.build_pv(&mut pv);

        let score = 1.0 - first_best_child.q();

        let time = start_time.elapsed().as_secs_f64();
        let [w, d, l] = [score, 0.0, 1.0 - score].map(|score| (score * 1000.0).round() as i32);
        let cp = score_to_cp(score);

        let capacity = self.tree.capacity();
        let filled = self.tree.len() as f64 / capacity as f64;

        print!(
            "info depth 1 seldepth 1 time {} nodes {} nps {} score cp {cp} wdl {w} {d} {l} hashfull {} pv",
            (time * 1000.0) as usize,
            iters,
            (iters as f64 / time) as usize,
            (filled * 1000.0) as usize
        );

        for &mv in pv.iter() {
            print!(" {}", mv);
        }

        println!();

        first_best_child
    }

    fn build_pv(&self, dst: &mut ArrayVec<Move, { MAX_DEPTH as usize }>) -> &Node {
        let mut node = &self.tree[0];

        let mut first_best_child = None;

        loop {
            let mut best_child_idx = 0;
            let mut best_child_q = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = &self.tree[child_idx];

                if child.visits == 0 {
                    continue;
                }

                let q = 1.0 - child.q();
                if q > best_child_q {
                    best_child_idx = child_idx;
                    best_child_q = q;
                }
            }

            if best_child_idx == 0 {
                assert!(first_best_child.is_some());
                break;
            }

            let best_child = &self.tree[best_child_idx];

            if first_best_child.is_none() {
                assert_ne!(best_child_idx, 0);
                first_best_child = Some(&self.tree[best_child_idx]);
            }

            if dst.try_push(best_child.mv.unwrap()).is_err() {
                break;
            }

            if best_child.child_count == 0 {
                break;
            }

            node = best_child;
        }

        first_best_child.unwrap()
    }

    fn construct_root(
        &mut self,
        root_pos: &Position,
        moves_to_search: &[Move],
        moves_to_exclude: &HashSet<Move>,
    ) {
        self.tree.push(Default::default());

        let mut moves = Vec::new();
        get_root_moves(&mut moves, root_pos, moves_to_search, moves_to_exclude);

        for &mv in moves.iter() {
            self.tree.push(Node::new(self.tree.len() as u32, 0, mv));
        }

        let root = &mut self.tree[0];
        root.first_child_idx = 1;
        root.child_count = moves.len() as u16;
    }

    #[must_use]
    fn select_expand(&mut self) -> Option<u32> {
        const CPUCT: f32 = 1.5;
        const FPU: f32 = 0.5;

        let mut curr = 0;
        loop {
            let node = &self.tree[curr];

            if curr != 0 {
                self.key_history.push(self.pos.key());
                self.pos.apply_move(node.mv.unwrap());
            }

            if node.is_terminal() || node.child_count == 0 {
                if !node.is_terminal() && !self.expand(curr as u32) {
                    return None;
                }
                break;
            }

            let e = CPUCT * (node.visits as f32).sqrt();
            let p = 1.0 / node.child_count as f32;

            let mut best_child = 0;
            let mut best_child_puct = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = &self.tree[child_idx];

                let q = if child.visits == 0 {
                    FPU
                } else {
                    1.0 - child.q()
                };
                let puct = q + e * p / (1 + child.visits) as f32;

                if puct > best_child_puct {
                    best_child = child_idx;
                    best_child_puct = puct;
                }
            }

            curr = best_child;
        }

        Some(curr as u32)
    }

    fn expand(&mut self, node_idx: u32) -> bool {
        assert!(!self.tree[node_idx as usize].is_terminal());

        if let Some(terminal_state) = check_terminal_state(
            &self.key_history,
            &self.pos,
            self.tree[node_idx as usize].mv.unwrap(),
        ) {
            let node = &mut self.tree[node_idx as usize];
            node.result = Some(terminal_state);
            return true;
        }

        let mut moves = Vec::new();
        generate_moves(&mut moves, &self.pos);

        if self.tree.len() + moves.len() > self.tree.capacity() {
            return false;
        }

        let first_child_idx = self.tree.len() as u32;

        for &mv in moves.iter() {
            self.tree
                .push(Node::new(self.tree.len() as u32, node_idx, mv));
        }

        let node = &mut self.tree[node_idx as usize];
        node.first_child_idx = first_child_idx;
        node.child_count = moves.len() as u16;

        true
    }

    #[must_use]
    fn simulate(&mut self, node_idx: u32) -> f32 {
        let node = &self.tree[node_idx as usize];
        node.result
            .map(TerminalState::u)
            .unwrap_or_else(|| eval_value(&self.pos))
    }

    fn backprop(&mut self, mut node_idx: u32, mut u: f32) {
        assert_ne!(node_idx, 0);
        loop {
            let node = &mut self.tree[node_idx as usize];

            node.visits += 1;
            node.total_score += u as f64;

            if node_idx == 0 {
                break;
            }

            node_idx = node.parent;
            u = 1.0 - u;
        }
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

impl TerminalState {
    #[must_use]
    fn u(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.5,
            Self::Loss => 0.0,
        }
    }
}

// returns wins/losses from the perspective of the *position after the move*
fn check_terminal_state(
    key_history: &[u64],
    pos: &Position,
    prev_move: Move,
) -> Option<TerminalState> {
    // player we care about: the one that made the move
    let stm = pos.stm().flip();

    if pos.has_road(stm) {
        return Some(TerminalState::Loss);
    }

    if pos.has_road(stm.flip()) {
        return Some(TerminalState::Win);
    }

    match pos.count_flats() {
        FlatCountOutcome::None => {}
        FlatCountOutcome::Draw => return Some(TerminalState::Draw),
        FlatCountOutcome::Win(color) => {
            return if color == stm {
                Some(TerminalState::Loss)
            } else {
                Some(TerminalState::Win)
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
