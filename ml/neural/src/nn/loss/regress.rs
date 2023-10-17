/*
    Appellation: regress <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Loss;

pub enum RegressiveLoss {
    Huber(HuberLoss),
    MeanAbsoluteError,
    MeanSquaredError,
    Other(String),
}

pub struct HuberLoss {
    delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn set_delta(&mut self, delta: f64) {
        self.delta = delta;
    }
}

impl Loss for HuberLoss {
    fn loss(&self, pred: &[f64], target: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (x, y) in pred.iter().zip(target.iter()) {
            let diff = x - y;
            if diff.abs() <= self.delta {
                // If the difference is sufficiently small, use the squared error.
                loss += 0.5 * diff.powi(2);
            } else {
                // Otherwise, use a variant of the absolute error.
                loss += self.delta * (diff.abs() - 0.5 * self.delta);
            }
        }
        loss / pred.len() as f64
    }
}
