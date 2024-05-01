/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod math;
pub mod predict;
pub mod propagate;
pub mod store;

pub mod arr {
    pub use self::generate::*;
    pub use self::{convert::*, like::*, ops::*};

    pub(crate) mod convert;
    pub(crate) mod generate;
    pub(crate) mod like;
    pub(crate) mod ops;
}

pub trait Initialize {
    fn init(&mut self);
}

pub trait Configure {
    type Config;

    fn setup(&mut self, config: Self::Config);
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub(crate) mod prelude {
    pub use super::arr::*;
    pub use super::math::*;
    pub use super::predict::*;
    pub use super::propagate::*;
    pub use super::store::*;
    pub use super::{Initialize, Transform};
}

#[cfg(test)]
mod tests {}
