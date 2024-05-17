/*
   Appellation: activate <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{binary::*, nl::*};

pub mod binary;
pub mod nl;

pub fn linear<T>(x: &T) -> T
where
    T: Clone,
{
    x.clone()
}

unary!(LinearActivation::linear(&self));

impl<T> LinearActivation for T
where
    T: Clone,
{
    type Output = T;

    fn linear(&self) -> Self::Output {
        linear(self)
    }
}

pub(crate) mod prelude {
    pub use super::binary::*;
    pub use super::nl::*;
    pub use super::{linear, LinearActivation};
}

#[doc(hidden)]
pub trait Evaluate<T> {
    type Output;

    fn eval(&self, args: T) -> Self::Output;
}
