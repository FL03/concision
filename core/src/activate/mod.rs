/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Activation functions for neural networks and their components. These functions are often
//! used to introduce non-linearity into the model, allowing it to learn more complex patterns
//! in the data.
//!
//! ## Overview
//!
//! This module works to provide a complete set of activation utilities for neural networks,
//! manifesting in a number of traits, utilities, and other primitives used to define various
//! approaches to activation functions.
//!
//! - [`HeavysideActivation`]
//! - [`LinearActivation`]
//! - [`SigmoidActivation`]
//! - [`SoftmaxActivation`]
//! - [`ReLUActivation`]
//! - [`TanhActivation`]
//!
#[doc(inline)]
pub use self::{traits::*, utils::*};

mod utils;

mod traits {
    #[doc(inline)]
    pub use self::{activate::*, activator::*, unary::*};

    mod activate;
    mod activator;
    mod unary;
}

mod impls {
    mod impl_binary;
    mod impl_linear;
    mod impl_nonlinear;
}

pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::utils::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_activation() {
        let linear = Linear;
        let input = 5.0;
        let output = linear.activate(input);
        assert_eq!(output, 5.0);

        let derivative = linear.activate_gradient(input);
        assert_eq!(derivative, 1.0);
    }
}
