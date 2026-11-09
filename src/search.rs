use crate::eval::policy::PolicyContext;
use crate::eval::value::eval_value;
use crate::limit::Limits;
use crate::movegen::{MoveList, generate_moves};
use crate::position::{FlatCountOutcome, Position};
use crate::takmove::Move;
use crate::tei::{TeiOptions, score_to_cp};
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
    policy: f32,
    visits: usize,
    total_score: f64,
    first_child_idx: u32,
    child_count: u16,
    state: Option<GameState>,
}

impl Node {
    #[must_use]
    fn new(idx: u32, parent: u32, mv: Move, policy: f32) -> Self {
        Self {
            idx,
            parent,
            mv: Some(mv),
            policy,
            ..Default::default()
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
        self.state.is_some_and(GameState::is_terminal)
    }
}

pub struct Searcher {
    options: TeiOptions,
    tree: Vec<Node>,
    pos: Position,
    key_history: Vec<u64>,
    iters: usize,
    selection_depth: i32,
}

impl Searcher {
    #[must_use]
    pub fn new(tree_size_mib: usize) -> Self {
        let node_count = tree_size_mib * 1024 * 1024 / size_of::<Node>();
        Self {
            options: Default::default(),
            tree: Vec::with_capacity(node_count),
            pos: Position::startpos(),
            key_history: Vec::with_capacity(1024),
            iters: 0,
            selection_depth: 0,
        }
    }

    pub fn set_tree_size(&mut self, size_mib: usize) {
        self.tree.clear();
        self.tree.shrink_to_fit();

        let node_count = size_mib * 1024 * 1024 / size_of::<Node>();
        self.tree = Vec::with_capacity(node_count);
    }

    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.iters
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_search(
        &mut self,
        root_pos: &Position,
        new_key_history: &[u64],
        start_time: Instant,
        limits: Limits,
        max_depth: i32,
        moves_to_search: &[Move],
        moves_to_exclude: &HashSet<Move>,
        options: &TeiOptions,
    ) {
        const MAX_REPORT_INTERVAL: usize = 32768;

        self.options = *options;

        self.tree.clear();
        self.construct_root(root_pos, moves_to_search, moves_to_exclude);

        self.key_history.clear();
        self.key_history.reserve(new_key_history.len() + MAX_DEPTH as usize);
        self.key_history.extend_from_slice(new_key_history);

        self.iters = 0;

        let mut depth_total = 0;
        let mut depth_max = 0;

        let mut average_depth = 0;

        let mut last_report_iter = 0;

        loop {
            let iter_idx = self.iters;

            self.pos = *root_pos;
            self.key_history.truncate(new_key_history.len());
            self.selection_depth = 0;

            let Some(leaf) = self.select_expand() else {
                if !options.silent && !options.minimal {
                    println!("info string Tree full, stopping");
                }
                break;
            };

            assert_ne!(leaf, 0);

            let u = self.simulate(leaf);
            self.backprop(leaf, u);

            self.iters += 1;

            depth_total += self.selection_depth as u64;
            depth_max = depth_max.max(self.selection_depth);

            let new_average_depth = (depth_total as f64 / self.iters as f64).round() as i32;
            let should_report_depth = new_average_depth > average_depth;

            average_depth = average_depth.max(new_average_depth);

            if average_depth >= max_depth || limits.should_stop(self.iters) {
                break;
            }

            if !options.silent
                && !options.minimal
                && (should_report_depth || (iter_idx - last_report_iter).is_multiple_of(MAX_REPORT_INTERVAL))
            {
                self.report(start_time, self.iters, average_depth, depth_max);
                last_report_iter = iter_idx;
            }
        }

        if !options.silent {
            self.final_report(start_time, self.iters, average_depth, depth_max);
        }
    }

    fn final_report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) {
        let first_best_child = self.report(start_time, iters, depth, seldepth);
        assert_ne!(first_best_child.idx, 0);

        if self.options.report_root_stats && !self.options.silent && !self.options.minimal {
            println!();
            let root = &self.tree[0];
            println!("info string root V: {} Q: {}", root.visits, root.q());
            for child_idx in self.tree[0].children() {
                let node = &self.tree[child_idx];
                let q = match node.visits > 0 {
                    true => (1.0 - node.q()).to_string(),
                    false => "n/a".to_string(),
                };
                println!(
                    "info string {} V: {}, Q: {} P: {}%",
                    node.mv.unwrap(),
                    node.visits,
                    q,
                    node.policy * 100.0
                );
            }
            println!();
        }

        let child_visits: usize = self.tree[0].children().map(|idx| self.tree[idx].visits).sum();
        assert_eq!(child_visits, self.tree[0].visits);

        println!("bestmove {}", first_best_child.mv.unwrap());
    }

    fn report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) -> &Node {
        let mut pv = ArrayVec::new();
        let first_best_child = self.build_pv(&mut pv);

        let score = 1.0 - first_best_child.q();

        let time = start_time.elapsed().as_secs_f64();
        let [w, d, l] = [score, 0.0, 1.0 - score].map(|score| (score * 1000.0).round() as i32);
        let cp = score_to_cp(score);

        let capacity = self.tree.capacity();
        let filled = self.tree.len() as f64 / capacity as f64;

        print!(
            "info depth {depth} seldepth {seldepth} time {} nodes {} nps {} score cp {cp} wdl {w} {d} {l} hashfull {} pv",
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

    fn construct_root(&mut self, root_pos: &Position, moves_to_search: &[Move], moves_to_exclude: &HashSet<Move>) {
        self.tree.push(Default::default());

        let mut moves = MoveList::new();
        get_root_moves(&mut moves, root_pos, moves_to_search, moves_to_exclude);

        let policy_ctx = PolicyContext::new(&root_pos);
        for &mv in moves.iter() {
            let policy = self.get_policy(&policy_ctx, root_pos, mv, true);
            self.tree.push(Node::new(self.tree.len() as u32, 0, mv, policy));
        }

        self.normalize_policy(1..self.tree.len());

        self.tree[1..].sort_by(|a, b| b.policy.partial_cmp(&a.policy).unwrap());

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
            if curr != 0 {
                self.key_history.push(self.pos.key());
                self.pos.apply_move(self.tree[curr].mv.unwrap());
            }

            if self.tree[curr].is_terminal() {
                break;
            }

            if curr != 0 && self.tree[curr].visits == 1 && !self.expand(curr as u32) {
                return None;
            }

            if self.tree[curr].child_count == 0 {
                break;
            }

            self.selection_depth += 1;

            let node = &self.tree[curr];

            let fpu = if curr == 0 { 1000.0 } else { FPU };
            let e = CPUCT * (node.visits as f32).sqrt();

            let mut best_child = 0;
            let mut best_child_puct = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = &self.tree[child_idx];

                let q = if child.visits == 0 { fpu } else { 1.0 - child.q() };
                let puct = q + e * child.policy / (1 + child.visits) as f32;

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
        assert_eq!(self.tree[node_idx as usize].child_count, 0);

        let mut moves = MoveList::new();
        generate_moves(&mut moves, &self.pos);

        if self.tree.len() + moves.len() > self.tree.capacity() {
            return false;
        }

        let first_child_idx = self.tree.len() as u32;

        let policy_ctx = PolicyContext::new(&self.pos);
        for &mv in moves.iter() {
            let policy = self.get_policy(&policy_ctx, &self.pos, mv, false);
            self.tree.push(Node::new(self.tree.len() as u32, node_idx, mv, policy));
        }

        self.normalize_policy(first_child_idx as usize..self.tree.len());

        let node = &mut self.tree[node_idx as usize];
        node.first_child_idx = first_child_idx;
        node.child_count = moves.len() as u16;

        true
    }

    #[must_use]
    fn simulate(&mut self, node_idx: u32) -> f32 {
        let node = &mut self.tree[node_idx as usize];

        if node.state.is_none() {
            let state = check_game_state(&self.key_history, &self.pos, node.mv.unwrap());
            node.state = Some(state);
        }

        node.state
            .and_then(GameState::u)
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

    fn get_policy(&self, ctx: &PolicyContext, pos: &Position, mv: Move, root: bool) -> f32 {
        const ROOT_PST: f32 = 3.0;
        const PST: f32 = 2.0;

        let mut new_pos = *pos;
        new_pos.apply_move(mv);

        if new_pos.has_road(pos.stm()) {
            return 500.0;
        } else if new_pos.has_road(pos.stm().flip()) {
            return -500.0;
        } else {
            match new_pos.count_flats() {
                FlatCountOutcome::Draw => return 0.0,
                FlatCountOutcome::Win(side) if side == pos.stm() => return 500.0,
                FlatCountOutcome::Win(side) if side == pos.stm().flip() => return -500.0,
                _ => {}
            }
        }

        let pst = if root { ROOT_PST } else { PST };

        if self.options.value_policy {
            let value = 1.0 - eval_value(&new_pos);
            4.0 * (value / (1.0 - value)).ln() / pst
        } else {
            ctx.get(mv) / pst
        }
    }

    fn normalize_policy(&mut self, nodes: Range<usize>) {
        let nodes = &mut self.tree[nodes];

        let max = nodes.iter().map(|node| node.policy).reduce(f32::max).unwrap();

        for node in nodes.iter_mut() {
            node.policy = (node.policy - max).exp();
        }

        let total: f32 = nodes.iter().map(|node| node.policy).sum();

        for node in nodes.iter_mut() {
            node.policy /= total;
        }
    }
}

fn get_root_moves(
    moves: &mut MoveList,
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

        moves.try_extend_from_slice(moves_to_search).unwrap();
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
enum GameState {
    Win,
    Draw,
    Loss,
    Ongoing,
}

impl GameState {
    #[must_use]
    fn is_terminal(self) -> bool {
        self != Self::Ongoing
    }

    #[must_use]
    fn u(self) -> Option<f32> {
        match self {
            Self::Win => Some(1.0),
            Self::Draw => Some(0.5),
            Self::Loss => Some(0.0),
            Self::Ongoing => None,
        }
    }
}

// returns wins/losses from the perspective of the *position after the move*
fn check_game_state(key_history: &[u64], pos: &Position, prev_move: Move) -> GameState {
    // player we care about: the one that made the move
    let stm = pos.stm().flip();

    if pos.has_road(stm) {
        return GameState::Loss;
    }

    if pos.has_road(stm.flip()) {
        return GameState::Win;
    }

    match pos.count_flats() {
        FlatCountOutcome::None => {}
        FlatCountOutcome::Draw => return GameState::Draw,
        FlatCountOutcome::Win(color) => {
            return if color == stm { GameState::Loss } else { GameState::Win };
        }
    }

    if prev_move.is_spread() {
        let mut repetitions = 0;
        for &key in key_history.iter().rev() {
            if key == pos.key() {
                repetitions += 1;
                if repetitions == 2 {
                    return GameState::Draw;
                }
            }
        }
    }

    GameState::Ongoing
}
