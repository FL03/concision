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
    use num::complex::Complex;

    #[test]
    fn test_ssm() {
        let step = 0.001;

        let config = SSMConfig::new(true, 9, 2);
        let model = SSM::<Complex<f64>>::create(config).init();
        assert!(model.discretize(step).is_ok());
    }
}
