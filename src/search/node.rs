use crate::search::GameState;
use crate::search::tree::TreeHalf;
use crate::takmove::Move;

#[cfg(feature = "large_tree_support")]
type IndexType = u64;
#[cfg(not(feature = "large_tree_support"))]
type IndexType = u32;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NodeIndex {
    inner: IndexType,
}

impl NodeIndex {
    const HALF_OFFSET: u32 = size_of::<IndexType>() as u32 * 8 - 1;

    const HALF_MASK: IndexType = 1 << Self::HALF_OFFSET;
    const IDX_MASK: IndexType = !Self::HALF_MASK;

    #[must_use]
    pub const fn new(half: TreeHalf, idx: u64) -> Self {
        debug_assert!((idx & !(Self::IDX_MASK as u64)) == 0);
        Self {
            inner: ((half as IndexType) << Self::HALF_OFFSET) | idx as IndexType,
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

pub struct NodeUpdateContext {
    score: u32,
    sq_score: u32,
    inv_sq_score: u32,
    d: u32,
}

impl NodeUpdateContext {
    pub fn new(score: f32, d: f32) -> Self {
        let sq_score = score.powi(2);
        let inv_sq_score = (1.0 - score).powi(2);
        Self {
            score: Self::quantize(score),
            sq_score: Self::quantize(sq_score),
            inv_sq_score: Self::quantize(inv_sq_score),
            d: Self::quantize(d),
        }
    }

    pub fn flip(&mut self) {
        self.score = (1u32 << 31) - self.score;
        std::mem::swap(&mut self.sq_score, &mut self.inv_sq_score);
    }

    fn quantize(score: f32) -> u32 {
        (score * (1u32 << 31) as f32).round() as u32
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Node {
    pub mv: Option<Move>,
    pub policy: f32,
    pub visits: u32,
    pub score: u32,
    pub sq_score: u32,
    pub d: u32,
    pub first_child_idx: NodeIndex,
    pub child_count: u16,
    pub state: Option<GameState>,
    policy_entropy: u8,
}

impl Node {
    #[must_use]
    pub fn new(mv: Move, policy: f32) -> Self {
        Self {
            mv: Some(mv),
            policy,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn new_root() -> Self {
        Self {
            mv: None,
            policy: 0.0,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn is_excluded(&self) -> bool {
        self.policy == f32::NEG_INFINITY
    }

    #[must_use]
    fn q_internal(&self) -> f64 {
        assert!(self.visits > 0);
        self.score as f64 / (self.visits << self.visits.leading_zeros()) as f64
    }

    #[must_use]
    fn sq_score_internal(&self) -> f64 {
        assert!(self.visits > 0);
        self.sq_score as f64 / (self.visits << self.visits.leading_zeros()) as f64
    }

    #[must_use]
    fn d_internal(&self) -> f64 {
        assert!(self.visits > 0);
        self.d as f64 / (self.visits << self.visits.leading_zeros()) as f64
    }

    #[must_use]
    pub fn q(&self) -> f32 {
        assert!(self.visits > 0);
        match self.state {
            Some(GameState::Win(_)) => 1.0,
            Some(GameState::Draw) => 0.5,
            Some(GameState::Loss(_)) => 0.0,
            _ => self.q_internal() as f32,
        }
    }

    #[must_use]
    pub fn q_variance(&self) -> f32 {
        assert!(self.visits > 0);
        let var = self.sq_score_internal() - self.q_internal().powi(2);
        var.max(0.0) as f32
    }

    #[must_use]
    pub fn d(&self) -> f32 {
        assert!(self.visits > 0);
        match self.state {
            Some(GameState::Win(_) | GameState::Loss(_)) => 0.0,
            Some(GameState::Draw) => 1.0,
            _ => self.d_internal() as f32,
        }
    }

    #[must_use]
    pub fn children(&self) -> ChildIterator {
        assert!(self.child_count > 0);
        let half = self.first_child_idx.half();
        let start = self.first_child_idx.idx() as u64;
        let end = start + self.child_count as u64;
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

    pub fn update(&mut self, ctx: &NodeUpdateContext) {
        self.visits += 1;

        let curr_score = if self.visits & (self.visits - 1) == 0 {
            (self.score + 1) >> 1
        } else {
            self.score
        };

        let curr_sq_score = if self.visits & (self.visits - 1) == 0 {
            (self.sq_score + 1) >> 1
        } else {
            self.sq_score
        };

        let curr_d = if self.visits & (self.visits - 1) == 0 {
            (self.d + 1) >> 1
        } else {
            self.d
        };

        let shift = self.visits.ilog2();

        self.score = curr_score + ((ctx.score + ((1 << shift) >> 1)) >> shift);
        self.sq_score = curr_sq_score + ((ctx.sq_score + ((1 << shift) >> 1)) >> shift);
        self.d = curr_d + ((ctx.d + ((1 << shift) >> 1)) >> shift);
    }

    pub fn set_state(&mut self, state: GameState) {
        self.state = Some(state);
    }
}

pub struct ChildIterator {
    half: TreeHalf,
    curr: u64,
    end: u64,
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
