/*
   Appellation: concision-ml <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-ml
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!

#[cfg(feature = "neural")]
pub use concision_neural as neural;
#[cfg(feature = "nlp")]
pub use concision_nlp as nlp;
#[cfg(feature = "optim")]
pub use concision_optim as optim;
#[cfg(feature = "transformers")]
pub use concision_transformers as transformers;

pub mod prelude {
    
    #[cfg(feature = "neural")]
    pub use concision_neural::prelude::*;
    #[cfg(feature = "nlp")]
    pub use concision_nlp::prelude::*;
    #[cfg(feature = "optim")]
    pub use concision_optim::prelude::*;
    #[cfg(feature = "transformers")]
    pub use concision_transformers::prelude::*;
}
