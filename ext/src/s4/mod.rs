/*
    appellation: s4 <module>
    authors: @FL03
*/
//! This library provides the sequential, structured state-space (S4) model implementation for the Concision framework.
//!
//! ## References
//!
//! - [Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2106.08084)
//! - [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
//!
#[doc(inline)]
pub use self::model::*;

mod model;

pub(crate) mod prelude {
    pub use super::model::*;
}
