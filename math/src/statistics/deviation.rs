/*
    Appellation: deviation <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::utils::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct StandardDeviation {
    pub deviation: f64,
    pub mean: f64,
    pub variance: f64,
}

impl StandardDeviation {
    pub fn new(x: &[f64]) -> StandardDeviation {
        let mean = mean(x);
        let variance = x.iter().map(|&x| x * x).sum::<f64>() / x.len() as f64 - mean * mean;
        let deviation = variance.sqrt();
        StandardDeviation {
            deviation,
            mean,
            variance,
        }
    }

    pub fn compute(&mut self, x: &[f64]) -> f64 {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let variance = x.iter().map(|&x| x * x).sum::<f64>() / x.len() as f64 - mean * mean;
        let deviation = variance.sqrt();
        self.deviation = deviation;
        self.mean = mean;
        self.variance = variance;
        deviation
    }

    pub fn deviation(&self) -> f64 {
        self.deviation
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn variance(&self) -> f64 {
        self.variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::prelude::{RoundTo, Statistics};

    #[test]
    fn test_std() {
        let x = vec![1.0, 2.0, 4.0, 9.0, 3.0, 4.0, 5.0];
        let sd = StandardDeviation::new(&x);

        assert_eq!(x.std().round_to(5), sd.deviation().round_to(5));
    }
}
