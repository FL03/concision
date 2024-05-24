/*
   Appellation: traits <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod num;
pub mod ops;
pub mod predict;
pub mod setup;
pub mod train;

pub mod arr {
    pub use self::prelude::*;

    mod create;
    mod misc;
    mod ops;
    mod reshape;
    mod tensor;

    pub(crate) mod prelude {
        pub use super::create::*;
        pub use super::misc::*;
        pub use super::ops::*;
        pub use super::reshape::*;
        pub use super::tensor::*;
    }
}

pub mod misc {
    pub mod adjust;
    #[doc(hidden)]
    pub mod container;
    #[doc(hidden)]
    pub mod sequential;
    #[doc(hidden)]
    pub mod store;
    pub mod toggle;

    pub(crate) mod prelude {
        pub use super::adjust::*;
        pub use super::container::*;
        pub use super::sequential::*;
        pub use super::store::*;
        pub use super::toggle::*;
    }
}

pub(crate) mod prelude {
    pub use super::arr::prelude::*;
    pub use super::misc::prelude::*;
    pub use super::num::*;
    pub use super::ops::*;
    pub use super::predict::*;
    pub use super::setup::*;
    pub use super::train::*;
}
