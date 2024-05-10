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

pub(crate) mod prelude {
    pub use super::linear::Linear;
}
