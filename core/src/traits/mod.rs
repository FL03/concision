/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod math;
pub mod predict;
pub mod store;

pub mod arr {
    pub use self::generate::*;
    pub use self::{like::*, ops::*};

    pub(crate) mod generate;
    pub(crate) mod like;
    pub(crate) mod ops;
}

use ndarray::prelude::{Array, Dimension};

pub trait Apply<T> {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
}

impl<T, D> Apply<T> for Array<T, D>
where
    D: Dimension,
{
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        self.map(f)
    }
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub(crate) mod prelude {
    pub use super::arr::*;
    pub use super::math::*;
    pub use super::predict::*;
    pub use super::store::*;
    pub use super::{Apply, Transform};
}

#[cfg(test)]
mod tests {}
