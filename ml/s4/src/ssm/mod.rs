/*
    Appellation: ssm <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # State Space Models (SSM)
//!
//!  
pub use self::{config::*, model::*};

pub(crate) mod config;
pub(crate) mod model;

pub trait StateSpace {
    fn features(&self) -> usize;

    fn scan(&self, step: f64) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::assert_ok;
    use crate::neural::prelude::Predict;

    use ndarray::prelude::*;
    // use num::complex::Complex;

    const FEATURES: usize = 4;
    const SAMPLES: usize = 100;
    const _STEP: f64 = 1e-4;

    #[test]
    fn test_ssm() {
        let u = Array::range(0.0, 1.0, 1.0 / SAMPLES as f64);
        let config = SSMConfig::new(true, FEATURES, SAMPLES);
        let model = SSMLayer::<f64>::create(config).unwrap();
        let pred = assert_ok(model.predict(&u));

        println!("{:?}", pred);
    }
}
