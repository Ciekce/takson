use crate::search::GameState;
use crate::search::tree::TreeHalf;
use crate::takmove::Move;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NodeIndex {
    inner: u32,
}

impl NodeIndex {
    const HALF_OFFSET: u32 = 31;
    const HALF_MASK: u32 = 1 << Self::HALF_OFFSET;

    #[must_use]
    pub const fn new(half: TreeHalf, idx: u32) -> Self {
        debug_assert!((idx & Self::HALF_MASK) == 0);
        Self {
            inner: ((half as u32) << Self::HALF_OFFSET) | idx,
        }
    }

    #[must_use]
    pub const fn half(self) -> TreeHalf {
        match self.inner >> Self::HALF_OFFSET {
            0 => TreeHalf::A,
            1 => TreeHalf::B,
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        (self.inner & !Self::HALF_MASK) as usize
    }

    #[must_use]
    pub const fn is_root(self) -> bool {
        self.inner & !Self::HALF_MASK == 0
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Node {
    pub parent: NodeIndex,
    pub mv: Option<Move>,
    pub policy: f32,
    pub visits: usize,
    pub total_score: f64,
    pub total_sq_score: f32,
    pub first_child_idx: NodeIndex,
    pub child_count: u16,
    pub state: Option<GameState>,
    policy_entropy: u8,
}

impl Node {
    #[must_use]
    pub fn new(parent: NodeIndex, mv: Move, policy: f32) -> Self {
        Self {
            parent,
            mv: Some(mv),
            policy,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn q(&self) -> f32 {
        assert!(self.visits > 0);
        match self.state {
            Some(GameState::Win(_)) => 1.0,
            Some(GameState::Draw) => 0.5,
            Some(GameState::Loss(_)) => 0.0,
            _ => (self.total_score / self.visits as f64) as f32,
        }
    }

    #[must_use]
    pub fn q_variance(&self) -> f32 {
        assert!(self.visits > 0);
        let visits = self.visits as f64;
        let var = self.total_sq_score as f64 / visits - (self.total_score / visits).powi(2);
        var.max(0.0) as f32
    }

    #[must_use]
    pub fn children(&self) -> ChildIterator {
        assert!(self.child_count > 0);
        let half = self.first_child_idx.half();
        let start = self.first_child_idx.idx() as u32;
        let end = start + self.child_count as u32;
        ChildIterator { half, curr: start, end }
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.state.is_some_and(GameState::is_terminal)
    }

    pub fn set_policy_entropy(&mut self, policy_entropy: f32) {
        self.policy_entropy = (policy_entropy * 255.0) as u8;
    }

    #[must_use]
    pub fn policy_entropy(&self) -> f32 {
        self.policy_entropy as f32 / 255.0
    }

    pub fn update(&mut self, score: f32) {
        self.visits += 1;
        self.total_score += score as f64;
        self.total_sq_score += score * score;
    }

    pub fn set_state(&mut self, state: GameState) {
        self.state = Some(state);
    }
}

pub struct ChildIterator {
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
