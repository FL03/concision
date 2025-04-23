/*
    Appellation: train <module>
    Contrib: @FL03
*/
//! This module implements various training mechanisms for neural networks. Here, implemented
//! trainers are lazily evaluated providing greater flexibility and performance.
pub use self::trainer::Trainer;

pub mod trainer;

pub(crate) mod impls {
    pub mod impl_config;
    pub mod impl_trainer;
}

pub(crate) mod prelude {
    pub use super::trainer::*;
}
