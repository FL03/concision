/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod ops;
pub mod params;
pub mod predict;
pub mod setup;
pub mod shape;
pub mod train;

pub mod arr {
    pub use self::prelude::*;

    mod create;
    mod misc;
    mod ops;
    mod reshape;

    pub(crate) mod prelude {
        pub use super::create::*;
        pub use super::misc::*;
        pub use super::ops::*;
        pub use super::reshape::*;
    }
}

pub mod misc {
    pub use self::prelude::*;

    pub(crate) mod adjust;
    #[doc(hidden)]
    pub(crate) mod sequential;
    #[doc(hidden)]
    pub(crate) mod store;
    pub(crate) mod toggle;

    pub(crate) mod prelude {
        pub use super::adjust::*;
        pub use super::sequential::*;
        pub use super::store::*;
        pub use super::toggle::*;
    }
}

pub(crate) mod prelude {
    pub use super::arr::prelude::*;
    pub use super::misc::prelude::*;
    pub use super::ops::*;
    pub use super::params::*;
    pub use super::predict::*;
    pub use super::setup::*;
    pub use super::shape::*;
    pub use super::train::*;
}
