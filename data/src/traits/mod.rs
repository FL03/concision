/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{data::*, records::*, shape::*};

pub mod records;
pub mod shape;

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

pub(crate) mod prelude {
    pub use super::data::prelude::*;
    pub use super::records::*;
    pub use super::shape::*;
}
