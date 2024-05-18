/*
   Appellation: fft <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{FftDirection, FftPlan};

pub struct Fft {
    direction: FftDirection,
    plan: FftPlan,
}

impl Fft {
    pub fn new(direction: FftDirection, plan: FftPlan) -> Self {
        Self { direction, plan }
    }

    pub const fn direction(&self) -> FftDirection {
        self.direction
    }

    pub const fn plan(&self) -> &FftPlan {
        &self.plan
    }
}
