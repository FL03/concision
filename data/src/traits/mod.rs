/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{data::*, ext::*, records::*};

pub mod build;
pub mod records;

#[doc(hidden)]
pub mod data {
    pub use self::{container::*, repr::*};

    pub(crate) mod container;
    pub(crate) mod repr;

    pub(crate) mod prelude {
        pub use super::container::*;
        pub use super::repr::*;
    }
}

pub mod ext {
    pub use self::{ndarray::*, ndtensor::*, ndview::*};

    pub(crate) mod ndarray;
    pub(crate) mod ndtensor;
    pub(crate) mod ndview;

    pub(crate) mod prelude {
        pub use super::ndarray::*;
        pub use super::ndtensor::*;
        pub use super::ndview::*;
    }
}

pub(crate) mod prelude {
    pub use super::data::prelude::*;
    pub use super::ext::prelude::*;
    pub use super::records::*;
}
