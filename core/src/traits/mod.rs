/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod math;
pub mod setup;
pub mod store;

pub mod arr {
    pub use self::{like::*, misc::*, ops::*};

    pub(crate) mod like;
    pub(crate) mod misc;
    pub(crate) mod ops;
}

pub mod nn {
    pub use self::prelude::*;

    pub mod model;
    pub mod module;
    pub mod predict;
    pub mod train;

    pub(crate) mod prelude {
        pub use super::model::*;
        pub use super::module::*;
        pub use super::predict::*;
        pub use super::train::*;
    }
}

pub trait Decrement {
    type Output;

    fn dec(&self) -> Self::Output;
}

pub trait Increment {
    type Output;

    fn inc(&self) -> Self::Output;
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

/*
 ******** implementations ********
*/
impl<D> Decrement for D
where
    D: nd::RemoveAxis,
{
    type Output = D::Smaller;

    fn dec(&self) -> Self::Output {
        self.remove_axis(nd::Axis(self.ndim() - 1))
    }
}

pub(crate) mod prelude {
    pub use super::{Decrement, Transform};

    pub use super::arr::*;
    pub use super::math::*;
    pub use super::nn::prelude::*;
    pub use super::setup::*;
    pub use super::store::*;
}

#[cfg(test)]
mod tests {}
