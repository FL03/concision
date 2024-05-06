/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{config::*, features::*, linear::*};

mod linear;

pub mod config;
pub mod features;

mod impls {
    pub mod impl_init;
    pub mod impl_linear;
    pub mod impl_model;
}

pub(crate) mod prelude {
    pub use super::config::Config as LinearConfig;
    pub use super::features::Features as LinearFeatures;
    pub use super::linear::Linear;
}
