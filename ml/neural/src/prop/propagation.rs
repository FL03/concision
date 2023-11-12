/*
    Appellation: propagation <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::PropagationMode;

pub struct Propagator {
    pub epochs: usize,
    pub mode: PropagationMode,
}

impl Propagator {
    pub fn new(epochs: usize, mode: PropagationMode) -> Self {
        Self { epochs, mode }
    }
}
