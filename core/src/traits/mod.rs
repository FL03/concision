/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod math;
pub mod predict;
pub mod setup;
pub mod store;
pub mod train;

pub mod arr {
    pub use self::generate::*;
    pub use self::{convert::*, like::*, ops::*};

    pub(crate) mod convert;
    pub(crate) mod generate;
    pub(crate) mod like;
    pub(crate) mod ops;
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub(crate) mod prelude {
    pub use super::Transform;

    pub use super::arr::*;
    pub use super::math::*;
    pub use super::predict::*;
    pub use super::setup::*;
    pub use super::store::*;
    pub use super::train::*;
}

#[cfg(test)]
mod tests {}
