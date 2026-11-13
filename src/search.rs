use crate::eval::value::eval_value;
use crate::limit::Limits;
use crate::movegen::MoveList;
use crate::position::{FlatCountOutcome, Position};
use crate::search::node::NodeIndex;
use crate::search::tree::Tree;
use crate::takmove::Move;
use crate::tei::{TeiOptions, score_to_cp};
use arrayvec::ArrayVec;
use std::collections::HashSet;
use std::time::Instant;

mod node;
mod tree;

pub const MAX_DEPTH: i32 = 255;

pub const DEFAULT_TREE_SIZE_MIB: usize = 16;
pub const MAX_TREE_SIZE_MIB: usize = 131072;

type PvList = ArrayVec<Move, { MAX_DEPTH as usize }>;

#[derive(Clone, Debug, Default)]
struct RootMove {
    pv: PvList,
    state: Option<GameState>,
    score: f32,
}

pub struct Searcher {
    options: TeiOptions,
    tree: Tree,
    root_pos: Position,
    pos: Position,
    key_history: Vec<u64>,
    root_key_history_len: usize,
    iters: usize,
    selection_depth: i32,
}

impl Searcher {
    #[must_use]
    pub fn new(tree_size_mib: usize) -> Self {
        Self {
            options: Default::default(),
            tree: Tree::new(tree_size_mib),
            root_pos: Position::startpos(),
            pos: Position::startpos(),
            key_history: Vec::with_capacity(1024),
            root_key_history_len: 0,
            iters: 0,
            selection_depth: 0,
        }
    }

    pub fn set_tree_size(&mut self, size_mib: usize) {
        self.tree.set_size(size_mib);
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

        self.root_pos = *root_pos;
        self.root_key_history_len = new_key_history.len();

        self.options = *options;

        self.key_history.clear();
        self.key_history.reserve(new_key_history.len() + MAX_DEPTH as usize);
        self.key_history.extend_from_slice(new_key_history);

        self.tree.init(options, root_pos, moves_to_search, moves_to_exclude);

        self.iters = 0;

        let mut depth_total = 0;
        let mut depth_max = 0;

        let mut average_depth = 0;

        let mut last_report_iter = 0;

        loop {
            let iter_idx = self.iters;

            self.reset_selection();

            let leaf = self.select_expand();

            assert!(!leaf.is_root());
            assert_eq!(leaf.half(), self.tree.active_half());

            let u = self.simulate(leaf);
            self.backprop(leaf, u);

            self.iters += 1;

            depth_total += self.selection_depth as u64;
            depth_max = depth_max.max(self.selection_depth);

            let new_average_depth = (depth_total as f64 / self.iters as f64).round() as i32;
            let should_report_depth = new_average_depth > average_depth;

            average_depth = average_depth.max(new_average_depth);

            if self.tree.root().is_terminal() {
                break;
            }

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

    pub fn reset(&mut self) {
        self.tree.reset();
    }

    fn final_report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) {
        let root_moves = self.report(start_time, iters, depth, seldepth);

        if self.options.report_root_stats && !self.options.silent && !self.options.minimal {
            println!();
            let root = self.tree.root();
            let state = match root.state {
                Some(state) if state != GameState::Ongoing => format!(" S: {:?}", state),
                _ => "".to_string(),
            };
            println!("info string root V: {} Q: {}{}", root.visits, root.q(), state);
            for child_idx in root.children() {
                let node = self.tree.node(child_idx);
                if node.policy == f32::NEG_INFINITY {
                    break;
                }
                let q = match node.visits > 0 {
                    true => (1.0 - node.q()).to_string(),
                    false => "n/a".to_string(),
                };
                let state = match node.state {
                    Some(state) if state != GameState::Ongoing => format!(" S: {:?}", state.flip()),
                    _ => "".to_string(),
                };
                println!(
                    "info string {} V: {}, Q: {} P: {}%{}",
                    node.mv.unwrap(),
                    node.visits,
                    q,
                    node.policy * 100.0,
                    state
                );
            }
            println!();
        }

        let child_visits: usize = self.tree.root().children().map(|idx| self.tree.node(idx).visits).sum();
        assert_eq!(child_visits, self.tree.root().visits);

        println!("bestmove {}", root_moves[0].pv[0]);
    }

    fn report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) -> Vec<RootMove> {
        let mut pvs = Vec::new();
        self.gather_pvs(&mut pvs);

        let time = start_time.elapsed().as_secs_f64();
        let hashfull = self.tree.filled_permille();

        for (pv_idx, root_move) in pvs.iter().enumerate() {
            let score = root_move.score;
            let [w, d, l] = [score, 0.0, 1.0 - score].map(|score| (score * 1000.0).round() as i32);

            let multipv = match self.options.multipv > 1 {
                true => format!(" multipv {}", pv_idx + 1),
                false => "".to_string(),
            };

            let score = match root_move.state {
                Some(GameState::Loss(plies)) if self.options.mate_scores => format!("mate {}", (plies + 2) / 2),
                Some(GameState::Win(plies)) if self.options.mate_scores => format!("mate {}", -(plies + 1) / 2),
                Some(GameState::Loss(plies)) if !self.options.mate_scores => format!("cp {}", 30000 - plies),
                Some(GameState::Win(plies)) if !self.options.mate_scores => format!("cp {}", -30000 + plies),
                _ => format!("cp {}", score_to_cp(score)),
            };

            print!(
                "info{multipv} depth {depth} seldepth {seldepth} time {} nodes {} nps {} score {score} wdl {w} {d} {l} hashfull {} pv",
                (time * 1000.0) as usize,
                iters,
                (iters as f64 / time) as usize,
                hashfull
            );

            for &mv in root_move.pv.iter() {
                print!(" {}", mv);
            }

            println!();
        }

        pvs
    }

    fn gather_pvs(&self, dst: &mut Vec<RootMove>) {
        let count = self.options.multipv.min(self.tree.root().child_count as usize);
        dst.reserve(count);

        let mut excluded = MoveList::new();

        for _ in 0..count {
            let mut root_move = RootMove::default();

            let mut node = self.tree.root();
            loop {
                let mut best_child_idx = self.tree.root_idx();
                let mut best_child_q = f32::NEG_INFINITY;

                for child_idx in node.children() {
                    let child = self.tree.node(child_idx);

                    if child.visits == 0 || child.policy == f32::NEG_INFINITY || excluded.contains(&child.mv.unwrap()) {
                        continue;
                    }

                    let q = match child.state {
                        Some(GameState::Win(plies)) => -1000.0 + plies as f32,
                        Some(GameState::Loss(plies)) => 1000.0 - plies as f32,
                        _ => 1.0 - child.q(),
                    };

                    if q > best_child_q {
                        best_child_idx = child_idx;
                        best_child_q = q;
                    }
                }

                if best_child_idx.is_root() {
                    break;
                }

                let best_child = self.tree.node(best_child_idx);

                if root_move.pv.is_empty() {
                    root_move.state = best_child.state;
                    root_move.score = 1.0 - best_child.q();
                }

                if root_move.pv.try_push(best_child.mv.unwrap()).is_err() {
                    break;
                }

                if best_child.child_count == 0 {
                    break;
                }

                node = best_child;
            }

            // not enough children have visits
            if root_move.pv.is_empty() {
                break;
            }

            excluded.push(root_move.pv[0]);
            dst.push(root_move);
        }
    }

    #[must_use]
    fn select_expand(&mut self) -> NodeIndex {
        const ROOT_CPUCT: f32 = 1.9;
        const CPUCT: f32 = 1.0;

        let mut curr = self.tree.root_idx();
        loop {
            if !curr.is_root() {
                self.key_history.push(self.pos.key());
                self.pos.apply_move(self.tree.node(curr).mv.unwrap());
            }

            if self.tree.node(curr).is_terminal() {
                break;
            }

            if !curr.is_root()
                && self.tree.node(curr).child_count == 0
                && self.tree.node(curr).visits > 0
                && !self.tree.expand(&self.pos, curr)
            {
                self.tree.flip();
                self.reset_selection();
                curr = self.tree.root_idx();
                continue;
            }

            if self.tree.node(curr).child_count == 0 {
                break;
            }

            if !self.tree.copy_children_over(curr) {
                self.tree.flip();
                self.reset_selection();
                curr = self.tree.root_idx();
                continue;
            }

            self.selection_depth += 1;

            let node = self.tree.node(curr);

            let cpuct = if curr.is_root() { ROOT_CPUCT } else { CPUCT };
            let fpu = if curr.is_root() { 1000.0 } else { node.q() };

            let e = {
                let mut e = cpuct;

                e *= (node.visits as f32).sqrt();
                e *= 1.0 + ((node.visits as f32 + 8192.0) / 8192.0).ln();
                e *= (0.44 - 1.44 * node.policy_entropy().max(0.0001).ln()).min(2.2);

                if node.visits > 1 {
                    e *= 1.0 + 0.85 * (node.q_variance().sqrt() * 5.0 - 1.0);
                }

                e
            };

            let mut proven_loss = true;
            let mut longest_loss = i16::MIN;

            let mut best_child = Default::default();
            let mut best_child_puct = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = self.tree.node(child_idx);

                if let Some(GameState::Win(plies)) = child.state {
                    longest_loss = longest_loss.max(plies);
                } else {
                    proven_loss = false;
                }

                let q = if child.visits == 0 { fpu } else { 1.0 - child.q() };
                let puct = q + e * child.policy / (1 + child.visits) as f32;

                if puct > best_child_puct {
                    best_child = child_idx;
                    best_child_puct = puct;
                }
            }

            if proven_loss {
                self.tree.node_mut(curr).set_state(GameState::Loss(longest_loss + 1));
            }

            curr = best_child;
        }

        assert!(!curr.is_root());

        curr
    }

    #[must_use]
    fn simulate(&mut self, node_idx: NodeIndex) -> f32 {
        let node = self.tree.node_mut(node_idx);

        if node.state.is_none() {
            let state = check_game_state(&self.key_history, &self.pos, node.mv.unwrap());
            node.set_state(state);
        }

        node.state
            .and_then(GameState::u)
            .unwrap_or_else(|| eval_value(&self.pos))
    }

    fn backprop(&mut self, mut node_idx: NodeIndex, mut u: f32) {
        assert!(!node_idx.is_root());
        let mut child_state = GameState::Ongoing;
        loop {
            assert_eq!(node_idx.half(), self.tree.active_half());

            let node = self.tree.node_mut(node_idx);

            if let GameState::Loss(plies) = child_state {
                assert!(!matches!(node.state, Some(GameState::Loss(_))));
                node.set_state(GameState::Win(plies + 1));
            }

            node.update(u);

            if node_idx.is_root() {
                break;
            }

            node_idx = node.parent;
            u = 1.0 - u;
            child_state = node.state.unwrap();
        }
    }

    fn reset_selection(&mut self) {
        self.pos = self.root_pos;
        self.key_history.truncate(self.root_key_history_len);
        self.selection_depth = 0;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum GameState {
    Win(i16),
    Draw,
    Loss(i16),
    Ongoing,
}

impl GameState {
    #[must_use]
    fn is_terminal(self) -> bool {
        matches!(self, Self::Win(0) | Self::Loss(0) | Self::Draw)
    }

    #[must_use]
    fn u(self) -> Option<f32> {
        match self {
            Self::Win(_) => Some(1.0),
            Self::Draw => Some(0.5),
            Self::Loss(_) => Some(0.0),
            Self::Ongoing => None,
        }
    }

    #[must_use]
    fn flip(self) -> Self {
        match self {
            Self::Win(plies) => Self::Loss(plies),
            Self::Loss(plies) => Self::Win(plies),
            _ => self,
        }
    }
}

// returns wins/losses from the perspective of the *position after the move*
fn check_game_state(key_history: &[u64], pos: &Position, prev_move: Move) -> GameState {
    // player we care about: the one that made the move
    let stm = pos.stm().flip();

    if pos.has_road(stm) {
        return GameState::Loss(0);
    }

    if pos.has_road(stm.flip()) {
        return GameState::Win(0);
    }

    match pos.count_flats() {
        FlatCountOutcome::None => {}
        FlatCountOutcome::Draw => return GameState::Draw,
        FlatCountOutcome::Win(color) => {
            return if color == stm {
                GameState::Loss(0)
            } else {
                GameState::Win(0)
            };
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
