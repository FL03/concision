/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!
#![crate_name = "concision_nn"]

pub use concision_neural::*;
#[cfg(feature = "nlp")]
pub use concision_nlp as nlp;
#[cfg(feature = "optim")]
pub use concision_optim as optim;
#[cfg(feature = "s4")]
pub use concision_s4 as s4;
#[cfg(feature = "transformers")]
pub use concision_transformers as transformers;

pub mod prelude {
    pub use concision_neural::prelude::*;
    #[cfg(feature = "nlp")]
    pub use concision_nlp::prelude::*;
    #[cfg(feature = "optim")]
    pub use concision_optim::prelude::*;
    #[cfg(feature = "s4")]
    pub use concision_s4::prelude::*;
    #[cfg(feature = "transformers")]
    pub use concision_transformers::prelude::*;
}
