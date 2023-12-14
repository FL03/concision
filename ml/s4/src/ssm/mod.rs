/*
    Appellation: ssm <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # State Space Models (SSM)
//!
//!  
pub use self::{config::*, model::*, params::*, utils::*};

pub(crate) mod config;
pub(crate) mod model;
pub(crate) mod params;

pub trait StateSpace {
    fn features(&self) -> usize;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm() {
        let step = 0.001;

        let config = SSMConfig::new(true, 9);
        let mut model = SSM::<f64>::create(config);
        assert!(model.descretize(step).is_ok());
    }
}
