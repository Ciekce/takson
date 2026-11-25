use crate::core::Color;
use crate::takmove::Move;

const DIVISOR: i32 = 8192;

pub struct History {
    entries: [[i16; 1 << Move::TOTAL_BITS]; Color::COUNT],
}

impl History {
    pub fn new() -> Self {
        Self {
            entries: [[0; 1 << Move::TOTAL_BITS]; Color::COUNT],
        }
    }

    pub fn clear(&mut self) {
        for a in &mut self.entries {
            a.fill(0);
        }
    }

    pub fn get(&self, stm: Color, mv: Move) -> f32 {
        self.entries[stm][mv.raw() as usize] as f32 / DIVISOR as f32
    }

    pub fn update(&mut self, stm: Color, mv: Move, bonus: i32) {
        let entry = &mut self.entries[stm][mv.raw() as usize];
        let value = i32::from(*entry);
        *entry = (value + bonus - value * bonus.abs() / DIVISOR) as i16;
    }
}

pub fn bonus(score: f32) -> i32 {
    let score = score.clamp(0.001, 0.999);
    (400.0 * (score / (1.0 - score)).ln()).round() as i32
}
