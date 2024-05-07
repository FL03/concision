/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Structured State Space Sequence Model (S4)
//!
//! ## Overview
//!
//! ## References
//!     - [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
pub use self::{config::*, module::*, params::*};

pub(crate) mod config;
pub(crate) mod module;
pub(crate) mod params;

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
        let config = S4Config::new(true, FEATURES, SAMPLES);
        let model = S4Layer::new(config).init().unwrap();
        let _pred = assert_ok(model.predict(&u));
    }
}
