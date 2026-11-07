use crate::core::Color;
use crate::position::Position;
use std::time::Instant;

pub const MAX_MOVE_OVERHEAD_MS: u64 = 10000;

const TIME_CHECK_INTERVAL: usize = 16;

const MAX_STONES_IN_HAND: u8 = 62;
const STONE_COMPLEXITY_SCALE: f64 = 5.0;

#[derive(Copy, Clone, Debug)]
struct TimeManager {
    max_time: f64,
}

impl TimeManager {
    fn new(remaining: f64, increment: f64, overhead: f64, pos: &Position) -> Self {
        let remaining = (remaining - overhead).max(0.001);

        let stones_in_hand = pos.stones_in_hand(Color::Black) + pos.stones_in_hand(Color::White);
        let stones_on_board = MAX_STONES_IN_HAND.saturating_sub(stones_in_hand);
        let complexity_offset = STONE_COMPLEXITY_SCALE * stones_on_board as f64 / MAX_STONES_IN_HAND as f64;
        let max_time = remaining * (17.5 + complexity_offset) / 400.0 + increment;

        let max_time = max_time.min(remaining * 0.8);

        Self { max_time }
    }

    fn should_stop(&self, _nodes: usize, time: f64) -> bool {
        time >= self.max_time
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Limits {
    start_time: Instant,
    nodes: Option<usize>,
    movetime: Option<f64>,
    time_manager: Option<TimeManager>,
}

impl Limits {
    pub fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            nodes: None,
            movetime: None,
            time_manager: None,
        }
    }

    pub fn set_nodes(&mut self, nodes: usize) -> bool {
        match self.nodes {
            None => {
                self.nodes = Some(nodes);
                true
            }
            Some(_) => false,
        }
    }

    pub fn set_movetime(&mut self, movetime: f64) -> bool {
        match self.movetime {
            None => {
                self.movetime = Some(movetime);
                true
            }
            Some(_) => false,
        }
    }

    pub fn set_time_manager(&mut self, remaining: f64, increment: f64, overhead: f64, pos: &Position) -> bool {
        match self.time_manager {
            None => {
                self.time_manager = Some(TimeManager::new(remaining, increment, overhead, pos));
                true
            }
            Some(_) => false,
        }
    }

    #[must_use]
    pub fn should_stop(&self, nodes: usize) -> bool {
        if let Some(max_nodes) = self.nodes
            && nodes >= max_nodes
        {
            return true;
        }

        if nodes > 0
            && nodes.is_multiple_of(TIME_CHECK_INTERVAL)
            && (self.movetime.is_some() || self.time_manager.is_some())
        {
            let time = self.start_time.elapsed().as_secs_f64();

            if let Some(movetime) = self.movetime
                && time >= movetime
            {
                return true;
            }

            if let Some(time_manager) = self.time_manager
                && time_manager.should_stop(nodes, time)
            {
                return true;
            }
        }

        false
    }
}
