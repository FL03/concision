/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! This module implements various activation functions for neural networks.
//!
//! ## Traits
//!
//! - [Heavyside]
//! - [LinearActivation]
//! - [Sigmoid]
//! - [Softmax]
//! - [ReLU]
//! - [Tanh]
//!
#[doc(inline)]
pub use self::prelude::*;

pub(crate) mod traits;

mod impls {
    mod impl_binary;
    mod impl_linear;
    mod impl_nonlinear;
}

pub(crate) mod prelude {
    pub use super::traits::*;
    pub use super::{Activate, ActivateGradient};
}

/// The [Activate] trait enables the definition of new activation functions often implemented
/// as _fieldless_ structs.
pub trait Activate<Rhs = Self> {
    type Output;

    fn activate(&self, rhs: Rhs) -> Self::Output;
}

pub trait ActivateGradient<Rhs = Self>: Activate<Self::Input> {
    type Input;
    type Delta;

    fn activate_gradient(&self, rhs: Rhs) -> Self::Delta;
}

/*
 ************* Implementations *************
*/

impl<X, Y> Activate<X> for Box<dyn Activate<X, Output = Y>> {
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self.as_ref().activate(rhs)
    }
}

impl<X, Y, F> Activate<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}
