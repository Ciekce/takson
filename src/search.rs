use crate::eval::policy::PolicyContext;
use crate::eval::value::eval_value;
use crate::limit::Limits;
use crate::movegen::{MoveList, generate_moves};
use crate::position::{FlatCountOutcome, Position};
use crate::takmove::Move;
use crate::tei::{TeiOptions, score_to_cp};
use arrayvec::ArrayVec;
use std::collections::HashSet;
use std::ops::{Index, IndexMut, Range};
use std::time::Instant;

pub const MAX_DEPTH: i32 = 255;

pub const DEFAULT_TREE_SIZE_MIB: usize = 16;
pub const MAX_TREE_SIZE_MIB: usize = 131072;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
enum TreeHalf {
    A = 0,
    B = 1,
}

impl TreeHalf {
    #[must_use]
    pub const fn root(self) -> NodeIndex {
        NodeIndex::new(self, 0)
    }

    pub const fn flip(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

impl<T> Index<TreeHalf> for [T; 2] {
    type Output = T;

    fn index(&self, index: TreeHalf) -> &Self::Output {
        // SAFETY: only implemented for arrays of the correct length
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<TreeHalf> for [T; 2] {
    fn index_mut(&mut self, index: TreeHalf) -> &mut Self::Output {
        // SAFETY: only implemented for arrays of the correct length
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct NodeIndex {
    inner: u32,
}

impl NodeIndex {
    const HALF_OFFSET: u32 = 31;
    const HALF_MASK: u32 = 1 << Self::HALF_OFFSET;

    #[must_use]
    const fn new(half: TreeHalf, idx: u32) -> Self {
        debug_assert!((idx & Self::HALF_MASK) == 0);
        Self {
            inner: ((half as u32) << Self::HALF_OFFSET) | idx,
        }
    }

    #[must_use]
    const fn half(self) -> TreeHalf {
        match self.inner >> Self::HALF_OFFSET {
            0 => TreeHalf::A,
            1 => TreeHalf::B,
            _ => unreachable!(),
        }
    }

    #[must_use]
    const fn idx(self) -> usize {
        (self.inner & !Self::HALF_MASK) as usize
    }

    #[must_use]
    const fn is_root(self) -> bool {
        self.inner & !Self::HALF_MASK == 0
    }
}

#[derive(Copy, Clone, Default, Debug)]
struct Node {
    idx: NodeIndex,
    parent: NodeIndex,
    mv: Option<Move>,
    policy: f32,
    visits: usize,
    total_score: f64,
    first_child_idx: NodeIndex,
    child_count: u16,
    state: Option<GameState>,
}

impl Node {
    #[must_use]
    fn new(idx: NodeIndex, parent: NodeIndex, mv: Move, policy: f32) -> Self {
        Self {
            idx,
            parent,
            mv: Some(mv),
            policy,
            ..Default::default()
        }
    }

    #[must_use]
    fn new_root(half: TreeHalf) -> Self {
        Self {
            idx: half.root(),
            parent: Default::default(),
            mv: None,
            policy: 0.0,
            ..Default::default()
        }
    }

    #[must_use]
    fn q(&self) -> f32 {
        assert!(self.visits > 0);
        match self.state {
            Some(GameState::Win(_)) => 1.0,
            Some(GameState::Draw) => 0.5,
            Some(GameState::Loss(_)) => 0.0,
            _ => (self.total_score / self.visits as f64) as f32,
        }
    }

    #[must_use]
    fn children(&self) -> ChildIterator {
        assert!(self.child_count > 0);
        let half = self.first_child_idx.half();
        let start = self.first_child_idx.idx() as u32;
        let end = start + self.child_count as u32;
        ChildIterator { half, curr: start, end }
    }

    #[must_use]
    fn is_terminal(&self) -> bool {
        self.state.is_some_and(GameState::is_terminal)
    }
}

struct ChildIterator {
    half: TreeHalf,
    curr: u32,
    end: u32,
}

impl Iterator for ChildIterator {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.end {
            let idx = NodeIndex::new(self.half, self.curr);
            self.curr += 1;
            Some(idx)
        } else {
            None
        }
    }
}

pub struct Searcher {
    options: TeiOptions,
    active_half: TreeHalf,
    tree: [Vec<Node>; 2],
    root_pos: Position,
    reused_tree: bool,
    pos: Position,
    key_history: Vec<u64>,
    root_key_history_len: usize,
    iters: usize,
    selection_depth: i32,
}

impl Searcher {
    #[must_use]
    pub fn new(tree_size_mib: usize) -> Self {
        let node_count = tree_size_mib * 1024 * 1024 / size_of::<Node>();
        Self {
            options: Default::default(),
            active_half: TreeHalf::A,
            tree: [Vec::with_capacity(node_count / 2), Vec::with_capacity(node_count / 2)],
            root_pos: Position::startpos(),
            reused_tree: false,
            pos: Position::startpos(),
            key_history: Vec::with_capacity(1024),
            root_key_history_len: 0,
            iters: 0,
            selection_depth: 0,
        }
    }

    #[must_use]
    fn get_free_node_idx(&self) -> NodeIndex {
        let idx = self.tree[self.active_half].len() as u32;
        NodeIndex::new(self.active_half, idx)
    }

    pub fn set_tree_size(&mut self, size_mib: usize) {
        for half in self.tree.iter_mut() {
            half.clear();
            half.shrink_to_fit();
        }

        let node_count = size_mib * 1024 * 1024 / size_of::<Node>();

        for half in self.tree.iter_mut() {
            *half = Vec::with_capacity(node_count / 2);
        }
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

        self.root_key_history_len = new_key_history.len();
        self.init_root(&root_pos, moves_to_search, moves_to_exclude);

        self.options = *options;

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

            self.reset_selection();

            let leaf = self.select_expand();

            assert!(!leaf.is_root());
            assert_eq!(leaf.half(), self.active_half);

            let u = self.simulate(leaf);
            self.backprop(leaf, u);

            self.iters += 1;

            depth_total += self.selection_depth as u64;
            depth_max = depth_max.max(self.selection_depth);

            let new_average_depth = (depth_total as f64 / self.iters as f64).round() as i32;
            let should_report_depth = new_average_depth > average_depth;

            average_depth = average_depth.max(new_average_depth);

            if self.tree[self.active_half][0].is_terminal() {
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
        for half in self.tree.iter_mut() {
            half.clear();
        }
    }

    fn final_report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) {
        let first_best_child = self.report(start_time, iters, depth, seldepth);
        assert!(!first_best_child.idx.is_root());

        if self.options.report_root_stats && !self.options.silent && !self.options.minimal {
            println!();
            let root = &self.tree[self.active_half][0];
            let state = match root.state {
                Some(state) if state != GameState::Ongoing => format!(" S: {:?}", state),
                _ => "".to_string(),
            };
            println!("info string root V: {} Q: {}{}", root.visits, root.q(), state);
            for child_idx in self.tree[self.active_half][0].children() {
                let node = &self.tree[self.active_half][child_idx.idx()];
                if node.policy == f32::NEG_INFINITY {
                    continue;
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

        if !self.reused_tree {
            let child_visits: usize = self.tree[self.active_half][0]
                .children()
                .map(|idx| self.tree[self.active_half][idx.idx()].visits)
                .sum();
            assert_eq!(child_visits, self.tree[self.active_half][0].visits);
        }

        println!("bestmove {}", first_best_child.mv.unwrap());
    }

    fn report(&self, start_time: Instant, iters: usize, depth: i32, seldepth: i32) -> &Node {
        let mut pv = ArrayVec::new();
        let first_best_child = self.build_pv(&mut pv);

        let score = 1.0 - first_best_child.q();

        let time = start_time.elapsed().as_secs_f64();
        let [w, d, l] = [score, 0.0, 1.0 - score].map(|score| (score * 1000.0).round() as i32);

        let capacity = self.tree[self.active_half].capacity();
        let filled = self.tree[self.active_half].len() as f64 / capacity as f64;

        let score = match first_best_child.state {
            Some(GameState::Loss(plies)) if self.options.mate_scores => format!("mate {}", (plies + 2) / 2),
            Some(GameState::Win(plies)) if self.options.mate_scores => format!("mate {}", -(plies + 1) / 2),
            Some(GameState::Loss(plies)) if !self.options.mate_scores => format!("cp {}", 30000 - plies),
            Some(GameState::Win(plies)) if !self.options.mate_scores => format!("cp {}", -30000 + plies),
            _ => format!("cp {}", score_to_cp(score)),
        };

        print!(
            "info depth {depth} seldepth {seldepth} time {} nodes {} nps {} score {score} wdl {w} {d} {l} hashfull {} pv",
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
        let mut node = &self.tree[self.active_half][0];

        let mut first_best_child = None;

        loop {
            let mut best_child_idx = self.active_half.root();
            let mut best_child_q = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = &self.tree[child_idx.half()][child_idx.idx()];

                if child.visits == 0 {
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
                assert!(first_best_child.is_some());
                break;
            }

            let best_child = &self.tree[best_child_idx.half()][best_child_idx.idx()];

            if first_best_child.is_none() {
                assert!(!best_child_idx.is_root());
                first_best_child = Some(&self.tree[best_child_idx.half()][best_child_idx.idx()]);
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

    fn init_root(&mut self, root_pos: &Position, moves_to_search: &[Move], moves_to_exclude: &HashSet<Move>) {
        let prev_pos = self.root_pos;
        if !self.try_find_root(&prev_pos, root_pos, self.active_half.root()) {
            self.reused_tree = false;

            for half in self.tree.iter_mut() {
                half.clear();
            }

            self.active_half = TreeHalf::A;

            let root_idx = self.active_half.root();
            self.tree[self.active_half].push(Node::new_root(self.active_half));

            let mut moves = MoveList::new();
            generate_moves(&mut moves, root_pos);

            for &mv in moves.iter() {
                let idx = self.get_free_node_idx();
                self.tree[self.active_half].push(Node::new(idx, root_idx, mv, 0.0));
            }

            let root = &mut self.tree[self.active_half][0];
            root.first_child_idx = NodeIndex::new(self.active_half, 1);
            root.child_count = moves.len() as u16;
        }

        let mut root_moves = MoveList::new();
        get_root_moves(&mut root_moves, root_pos, moves_to_search, moves_to_exclude);

        let policy_ctx = PolicyContext::new(root_pos);
        for child_idx in self.tree[self.active_half][0].children() {
            let mv = self.tree[self.active_half][child_idx.idx()].mv.unwrap();
            let policy = match root_moves.contains(&mv) {
                true => self.get_policy(&policy_ctx, root_pos, mv, true),
                false => f32::NEG_INFINITY,
            };
            self.tree[self.active_half][child_idx.idx()].policy = policy;
        }

        let root_child_count = self.tree[self.active_half][0].child_count as usize;
        self.tree[self.active_half][1..root_child_count + 1].sort_by(|a, b| b.policy.partial_cmp(&a.policy).unwrap());

        let unexcluded_end = (1..root_child_count + 1)
            .find(|&child_idx| self.tree[self.active_half][child_idx].policy == f32::NEG_INFINITY)
            .unwrap_or(root_child_count)
            + 1;

        self.normalize_policy(1..unexcluded_end);

        self.root_pos = *root_pos;
    }

    fn try_find_root(&mut self, prev_pos: &Position, root_pos: &Position, start: NodeIndex) -> bool {
        if self.tree[self.active_half].is_empty() {
            println!("info string no tree to reuse");
            return false;
        }

        if *prev_pos == *root_pos {
            println!("info string root unchanged");
            return true;
        }

        for child_idx in self.tree[start.half()][start.idx()].children() {
            let child = self.tree[child_idx.half()][child_idx.idx()];

            let mut new_pos = *prev_pos;
            new_pos.apply_move(child.mv.unwrap());

            if new_pos == *root_pos {
                if child.child_count == 0 {
                    println!("info string found unexpanded root :c");
                    return false;
                }

                println!("info string found new root?");

                self.tree[self.active_half][0] = child;

                self.tree[self.active_half][0].idx = self.active_half.root();
                self.tree[self.active_half][0].parent = Default::default();
                self.tree[self.active_half][0].mv = None;
                self.tree[self.active_half][0].state = None;

                let mut next_idx = 1;
                for subchild_idx in child.children() {
                    let mut subchild = self.tree[subchild_idx.half()][subchild_idx.idx()];
                    subchild.idx = NodeIndex::new(self.active_half, next_idx as u32);
                    subchild.parent = self.active_half.root();
                    self.tree[self.active_half][next_idx] = subchild;
                    next_idx += 1;
                }

                self.tree[self.active_half][0].first_child_idx = NodeIndex::new(self.active_half, 1);

                self.reused_tree = true;

                return true;
            }

            if start.is_root() && child.child_count > 0 && self.try_find_root(&new_pos, root_pos, child_idx) {
                return true;
            }
        }

        if start.is_root() {
            println!("info string failed to find usable tree");
        }

        false
    }

    #[must_use]
    fn select_expand(&mut self) -> NodeIndex {
        const ROOT_CPUCT: f32 = 1.9;
        const CPUCT: f32 = 1.5;

        let mut curr = self.active_half.root();
        loop {
            if curr.half() != self.active_half {
                eprintln!("waaagh");
                eprintln!("tree reused: {}", self.reused_tree);
                eprintln!("selection depth: {}", self.selection_depth);
            }

            assert_eq!(curr.half(), self.active_half);

            if !curr.is_root() {
                self.key_history.push(self.pos.key());
                self.pos.apply_move(self.tree[self.active_half][curr.idx()].mv.unwrap());
            }

            if self.tree[self.active_half][curr.idx()].is_terminal() {
                break;
            }

            if !curr.is_root()
                && self.tree[self.active_half][curr.idx()].child_count == 0
                && self.tree[self.active_half][curr.idx()].visits > 0
                && !self.expand(curr)
            {
                self.flip();
                self.reset_selection();
                curr = self.active_half.root();
                continue;
            }

            if self.tree[self.active_half][curr.idx()].child_count == 0 {
                break;
            }

            if !self.copy_children_over(curr) {
                self.flip();
                self.reset_selection();
                curr = self.active_half.root();
                continue;
            }

            self.selection_depth += 1;

            let node = &self.tree[self.active_half][curr.idx()];

            let cpuct = if curr.is_root() { ROOT_CPUCT } else { CPUCT };
            let fpu = if curr.is_root() { 1000.0 } else { node.q() };

            let e = cpuct * (node.visits as f32).sqrt();

            let mut proven_loss = true;
            let mut longest_loss = i16::MIN;

            let mut best_child = Default::default();
            let mut best_child_puct = f32::NEG_INFINITY;

            for child_idx in node.children() {
                let child = &self.tree[self.active_half][child_idx.idx()];

                if curr.is_root() && child.policy < 0.0 {
                    continue;
                }

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
                self.tree[self.active_half][curr.idx()].state = Some(GameState::Loss(longest_loss + 1));
            }

            curr = best_child;
        }

        assert!(!curr.is_root());

        curr
    }

    fn expand(&mut self, node_idx: NodeIndex) -> bool {
        assert!(!self.tree[self.active_half][node_idx.idx()].is_terminal());
        assert_eq!(self.tree[self.active_half][node_idx.idx()].child_count, 0);

        let mut moves = MoveList::new();
        generate_moves(&mut moves, &self.pos);

        if self.tree[self.active_half].len() + moves.len() > self.tree[self.active_half].capacity() {
            return false;
        }

        let first_child_idx = self.get_free_node_idx();

        let policy_ctx = PolicyContext::new(&self.pos);
        for &mv in moves.iter() {
            let idx = self.get_free_node_idx();
            let policy = self.get_policy(&policy_ctx, &self.pos, mv, false);
            self.tree[self.active_half].push(Node::new(idx, node_idx, mv, policy));
        }

        self.normalize_policy(first_child_idx.idx()..self.tree[self.active_half].len());

        let node = &mut self.tree[self.active_half][node_idx.idx()];
        node.first_child_idx = first_child_idx;
        node.child_count = moves.len() as u16;

        true
    }

    #[must_use]
    fn simulate(&mut self, node_idx: NodeIndex) -> f32 {
        let node = &mut self.tree[self.active_half][node_idx.idx()];

        if node.state.is_none() {
            let state = check_game_state(&self.key_history, &self.pos, node.mv.unwrap());
            node.state = Some(state);
        }

        node.state
            .and_then(GameState::u)
            .unwrap_or_else(|| eval_value(&self.pos))
    }

    fn backprop(&mut self, mut node_idx: NodeIndex, mut u: f32) {
        assert!(!node_idx.is_root());
        let mut child_state = GameState::Ongoing;
        loop {
            assert_eq!(node_idx.half(), self.active_half);
            let node = &mut self.tree[self.active_half][node_idx.idx()];

            if let GameState::Loss(plies) = child_state {
                assert!(!matches!(node.state, Some(GameState::Loss(_))));
                node.state = Some(GameState::Win(plies + 1));
            }

            node.visits += 1;
            node.total_score += u as f64;

            if node_idx.is_root() {
                break;
            }

            assert_ne!(node_idx, node.parent);

            node_idx = node.parent;
            u = 1.0 - u;
            child_state = node.state.unwrap();
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
        let nodes = &mut self.tree[self.active_half][nodes];

        let max = nodes.iter().map(|node| node.policy).reduce(f32::max).unwrap();

        for node in nodes.iter_mut() {
            node.policy = (node.policy - max).exp();
        }

        let total: f32 = nodes.iter().map(|node| node.policy).sum();

        for node in nodes.iter_mut() {
            node.policy /= total;
        }
    }

    fn flip(&mut self) {
        for node in self.tree[self.active_half].iter_mut() {
            if node.first_child_idx.half() != self.active_half {
                node.child_count = 0;
            }
        }

        self.active_half = self.active_half.flip();

        let root = self.tree[self.active_half.flip()][0];

        self.tree[self.active_half].clear();
        self.tree[self.active_half].push(root);
    }

    fn reset_selection(&mut self) {
        self.pos = self.root_pos;
        self.key_history.truncate(self.root_key_history_len);
        self.selection_depth = 0;
    }

    fn copy_children_over(&mut self, node_idx: NodeIndex) -> bool {
        assert_eq!(node_idx.half(), self.active_half);

        let node = &self.tree[self.active_half][node_idx.idx()];
        if node.first_child_idx.half() == self.active_half {
            return true;
        }

        if self.tree[self.active_half].len() + node.child_count as usize > self.tree[self.active_half].capacity() {
            return false;
        }

        let first_child_idx = self.get_free_node_idx();

        for child_idx in node.children() {
            let mut node = self.tree[child_idx.half()][child_idx.idx()];
            node.parent = node_idx;
            node.idx = self.get_free_node_idx();
            self.tree[self.active_half].push(node);
        }

        self.tree[self.active_half][node_idx.idx()].first_child_idx = first_child_idx;

        true
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
