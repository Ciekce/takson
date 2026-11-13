use crate::eval::policy::PolicyContext;
use crate::eval::value::eval_value;
use crate::movegen::{MoveList, generate_moves};
use crate::position::{FlatCountOutcome, Position};
use crate::search::node::{Node, NodeIndex};
use crate::takmove::Move;
use crate::tei::TeiOptions;
use std::collections::HashSet;
use std::ops::{Index, IndexMut, Range};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TreeHalf {
    A = 0,
    B = 1,
}

impl TreeHalf {
    #[must_use]
    pub const fn root(self) -> NodeIndex {
        NodeIndex::new(self, 0)
    }

    #[must_use]
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

#[must_use]
fn calc_node_count(mib: usize) -> usize {
    mib * 1024 * 1024 / size_of::<Node>()
}

pub struct Tree {
    active_half: TreeHalf,
    tree: [Vec<Node>; 2],
    value_policy: bool,
}

impl Tree {
    #[must_use]
    pub fn new(size_mib: usize) -> Self {
        let size = calc_node_count(size_mib) / 2;
        println!("half size: {}", size);
        Self {
            active_half: TreeHalf::A,
            tree: [Vec::with_capacity(size), Vec::with_capacity(size)],
            value_policy: false,
        }
    }

    pub fn set_size(&mut self, size_mib: usize) {
        for half in self.tree.iter_mut() {
            half.clear();
            half.shrink_to_fit();
        }

        let size = calc_node_count(size_mib) / 2;
        println!("half size: {}", size);

        for half in self.tree.iter_mut() {
            *half = Vec::with_capacity(size);
        }
    }

    pub fn reset(&mut self) {
        self.active_half = TreeHalf::A;
        for half in self.tree.iter_mut() {
            half.clear();
        }
    }

    pub fn init(
        &mut self,
        options: &TeiOptions,
        root_pos: &Position,
        moves_to_search: &[Move],
        moves_to_exclude: &HashSet<Move>,
    ) {
        self.reset();

        self.value_policy = options.value_policy;

        self.tree[self.active_half].push(Node::new_root());

        let root_idx = self.root_idx();

        let mut moves = MoveList::new();
        get_root_moves(&mut moves, root_pos, moves_to_search, moves_to_exclude);

        let policy_ctx = PolicyContext::new(root_pos);
        for &mv in moves.iter() {
            let policy = self.get_policy(&policy_ctx, root_pos, mv, true);
            self.tree[self.active_half].push(Node::new(root_idx, mv, policy));
        }

        let policy_entropy = self.normalize_policy(1..self.tree[self.active_half].len());

        self.tree[self.active_half][1..].sort_by(|a, b| b.policy.partial_cmp(&a.policy).unwrap());

        let active_half = self.active_half;

        let root = self.root_mut();
        root.first_child_idx = NodeIndex::new(active_half, 1);
        root.child_count = moves.len() as u16;

        self.root_mut().set_policy_entropy(policy_entropy);
    }

    pub fn flip(&mut self) {
        for node in self.tree[self.active_half].iter_mut() {
            if node.first_child_idx.half() != self.active_half {
                node.child_count = 0;
            }
        }

        let root = *self.root();

        self.active_half = self.active_half.flip();

        self.tree[self.active_half].clear();
        self.tree[self.active_half].push(root);
    }

    #[must_use]
    pub fn active_half(&self) -> TreeHalf {
        self.active_half
    }

    #[must_use]
    pub fn root_idx(&self) -> NodeIndex {
        self.active_half.root()
    }

    #[must_use]
    pub fn root(&self) -> &Node {
        &self.tree[self.active_half][0]
    }

    #[must_use]
    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.tree[self.active_half][0]
    }

    #[must_use]
    pub fn node(&self, idx: NodeIndex) -> &Node {
        &self.tree[idx.half()][idx.idx()]
    }

    #[must_use]
    pub fn node_mut(&mut self, idx: NodeIndex) -> &mut Node {
        &mut self.tree[idx.half()][idx.idx()]
    }

    pub fn expand(&mut self, pos: &Position, node_idx: NodeIndex) -> bool {
        assert!(!self.node(node_idx).is_terminal());
        assert_eq!(self.node(node_idx).child_count, 0);

        let mut moves = MoveList::new();
        generate_moves(&mut moves, pos);

        if self.tree[self.active_half].len() + moves.len() > self.tree[self.active_half].capacity() {
            return false;
        }

        let first_child_idx = self.next_free_node_idx();

        let policy_ctx = PolicyContext::new(pos);
        for &mv in moves.iter() {
            let policy = self.get_policy(&policy_ctx, pos, mv, false);
            self.tree[self.active_half].push(Node::new(node_idx, mv, policy));
        }

        let policy_entropy = self.normalize_policy(first_child_idx.idx()..self.tree[self.active_half].len());

        let node = self.node_mut(node_idx);

        node.first_child_idx = first_child_idx;
        node.child_count = moves.len() as u16;

        node.set_policy_entropy(policy_entropy);

        true
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

        if self.value_policy {
            let value = 1.0 - eval_value(&new_pos);
            4.0 * (value / (1.0 - value)).ln() / pst
        } else {
            ctx.get(mv) / pst
        }
    }

    fn normalize_policy(&mut self, nodes: Range<usize>) -> f32 {
        let nodes = &mut self.tree[self.active_half][nodes];

        let max = nodes.iter().map(|node| node.policy).reduce(f32::max).unwrap();

        let mut total = 0.0;

        for node in nodes.iter_mut() {
            node.policy = (node.policy - max).exp();
            total += node.policy;
        }

        let mut sum_of_squares = 0.0;

        for node in nodes.iter_mut() {
            node.policy /= total;
            sum_of_squares += node.policy * node.policy;
        }

        (1.0 - sum_of_squares).clamp(0.0, 1.0)
    }

    pub fn copy_children_over(&mut self, node_idx: NodeIndex) -> bool {
        assert_eq!(node_idx.half(), self.active_half);

        let node = self.node(node_idx);
        if node.first_child_idx.half() == self.active_half {
            return true;
        }

        if self.tree[self.active_half].len() + node.child_count as usize > self.tree[self.active_half].capacity() {
            return false;
        }

        let first_child_idx = self.next_free_node_idx();

        for child_idx in node.children() {
            let mut node = *self.node(child_idx);
            node.parent = node_idx;
            self.tree[self.active_half].push(node);
        }

        self.node_mut(node_idx).first_child_idx = first_child_idx;

        true
    }

    #[must_use]
    pub fn filled_permille(&self) -> usize {
        let capacity = self.tree[self.active_half].capacity();
        let filled = self.tree[self.active_half].len() as f64 / capacity as f64;
        (filled * 1000.0) as usize
    }

    #[must_use]
    fn next_free_node_idx(&self) -> NodeIndex {
        let idx = self.tree[self.active_half].len() as u32;
        NodeIndex::new(self.active_half, idx)
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
