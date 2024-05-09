/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::layout::prelude::*;
pub use self::{config::Config, linear::Linear};

mod linear;

pub mod config;

pub mod layout {
    pub use self::{features::*, layout::*};

    mod features;
    mod layout;

    pub(crate) mod prelude {
        pub use super::features::Features;
        pub use super::layout::Layout;
    }
}

mod impls {
    pub mod impl_init;
    pub mod impl_linear;
    pub mod impl_model;
}

pub(crate) mod prelude {
    pub use super::linear::Linear;
}
