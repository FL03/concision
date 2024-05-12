/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod generator;
pub mod num;
pub mod ops;
pub mod predict;
pub mod train;

pub mod arr {
    pub use self::prelude::*;

    pub(crate) mod create;
    pub(crate) mod misc;
    pub(crate) mod ops;
    pub(crate) mod shape;

    pub(crate) mod prelude {
        pub use super::create::*;
        pub use super::misc::*;
        pub use super::ops::*;
        pub use super::shape::*;
    }
}

pub(crate) mod misc {
    pub mod adjust;
    #[doc(hidden)]
    pub mod container;
    pub mod setup;
    pub mod store;

    pub(crate) mod prelude {
        pub use super::adjust::*;
        pub use super::container::*;
        pub use super::setup::*;
        pub use super::store::*;
    }
}

pub(crate) mod prelude {
    pub use super::arr::prelude::*;
    pub use super::generator::*;
    pub use super::misc::prelude::*;
    pub use super::num::*;
    pub use super::ops::*;
    pub use super::predict::*;
    pub use super::train::*;
}
