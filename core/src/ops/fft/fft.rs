/*
   Appellation: fft <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{FftDirection, FftPlan};

pub struct FastFourierTransform {
    direction: FftDirection,
    plan: FftPlan,
}

impl FastFourierTransform {
    pub fn new(direction: FftDirection, plan: FftPlan) -> Self {
        Self { direction, plan }
    }

    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    pub fn plan(&self) -> &FftPlan {
        &self.plan
    }
}
