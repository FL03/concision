/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod adjust;
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

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub(crate) mod prelude {
    pub use super::Transform;

    pub use super::adjust::*;
    pub use super::arr::*;
    pub use super::math::*;
    pub use super::nn::prelude::*;
    pub use super::setup::*;
    pub use super::store::*;
}

#[cfg(test)]
mod tests {}
