/*
    Appellation: kan <module>
    Created At: 2025.11.26:13:58:50
    Contrib: @FL03
*/
//! This library provides an implementation of the Kolmogorov–Arnold Networks (kan) model using
//! the [`concision`](https://docs.rs/concision) framework.
//!
//! ## References
//!
//! - [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v1)
//!
#[doc(inline)]
pub use self::model::*;

mod model;

pub(crate) mod prelude {
    pub use super::model::*;
}
