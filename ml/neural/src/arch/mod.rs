/*
    Appellation: arch <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Architecture
//!
//! This module describes the architecture of various components of the neural network.
pub use self::{architecture::*, deep::*, shallow::*, utils::*};

pub(crate) mod architecture;
pub(crate) mod deep;
pub(crate) mod shallow;

pub trait Arch {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stack::Stack;
    use crate::prelude::{Activate, Layer, LayerShape, Softmax};

    fn _stack<A: Activate<f64> + Default>(
        shapes: impl IntoIterator<Item = (usize, usize)>,
    ) -> Stack<f64, A> {
        let mut stack = Stack::new();
        for (inputs, outputs) in shapes.into_iter() {
            stack.push(Layer::<f64, A>::from(LayerShape::new(inputs, outputs)).init(true));
        }
        stack
    }

    #[test]
    fn test_arch() {
        assert!(true);
    }

    #[test]
    fn test_deep_network() {
        assert!(true);
    }
}
