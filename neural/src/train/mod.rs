/*
    appellation: train <module>
    authors: @FL03
*/
//! This module focuses on developing lazy trainers for neural networks.
#[doc(inline)]
pub use self::{error::*, trainer::Trainer, traits::*};

pub mod error;
/// this module provides the [`Trainer`] implementation, which is a generic, lazy trainer
/// for neural networks. It is designed to be flexible and extensible, allowing for the
/// implementation of various training algorithms and strategies. The trainer is built on top
/// of the [`Train`] trait, which defines the core training functionality. The trainer can be
/// used to train neural networks with different configurations and parameters, making it a
/// versatile tool for neural network training.
pub mod trainer;

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod train;
    mod trainers;

    mod prelude {
        #[doc(inline)]
        pub use super::train::*;
        #[doc(inline)]
        pub use super::trainers::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::trainer::*;
    #[doc(inline)]
    pub use super::traits::*;
}
