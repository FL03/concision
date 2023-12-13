/*
    Appellation: ssm <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # State Space Models (SSM)
//!
//!  
pub use self::{config::*, model::*, utils::*};

pub(crate) mod config;
pub(crate) mod model;

pub trait StateSpace {
    fn features(&self) -> usize;


}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm() {
        let model = SSM::create(9);
    }
}