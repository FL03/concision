/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{binary::*, linear::*, nl::*};

pub mod binary;
pub mod linear;
pub mod nl;

pub(crate) mod prelude {
    pub use super::binary::*;
    pub use super::linear::*;
    pub use super::nl::*;
    pub use super::{Activate, Evaluate};
}

#[doc(hidden)]
pub trait Activate<T> {
    type Output;

    fn activate(&self, args: &T) -> Self::Output;
}

#[doc(hidden)]
pub trait Evaluate<T> {
    type Output;

    fn eval(&self, args: T) -> Self::Output;
}

activator!(LinearActor::<T>(T::clone) where T: Clone);
