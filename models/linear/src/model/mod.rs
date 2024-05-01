/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{config::Config, features::Features, linear::Linear, params::LinearParams};

mod linear;

pub mod config;
pub mod features;
pub mod params;

pub(crate) mod prelude {
    pub use super::config::Config as LinearConfig;
    pub use super::features::Features as LinearFeatures;
    pub use super::linear::Linear;
    pub use super::params::LinearParams;
}
