/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::utils::*;
pub use self::{binary::*, linear::*, nonlinear::*};

pub(crate) mod utils;

pub mod binary;
pub mod linear;
pub mod nonlinear;

pub(crate) mod prelude {
    pub use super::binary::*;
    pub use super::linear::*;
    pub use super::nonlinear::*;
    pub use super::utils::*;
    pub use super::{Activate, Evaluate};
}

/// [Activate] designates a function or structure that can be used
/// as an activation function for a neural network.
///
/// The trait enables implemented models to employ various activation
/// functions either as a pure function or as a structure.
pub trait Activate<T> {
    type Output;

    fn activate(&self, args: T) -> Self::Output;
}

/// [Evaluate] is used for _lazy_, structured functions that evaluate to
/// some value.
pub trait Evaluate {
    type Output;

    fn eval(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/

activator!(LinearActor::<T>(T::clone) where T: Clone);

impl<F, U, V> Activate<U> for F
where
    F: Fn(U) -> V,
{
    type Output = V;

    fn activate(&self, args: U) -> Self::Output {
        self(args)
    }
}

impl<U, V> Activate<U> for Box<dyn Activate<U, Output = V>> {
    type Output = V;

    fn activate(&self, args: U) -> Self::Output {
        self.as_ref().activate(args)
    }
}
