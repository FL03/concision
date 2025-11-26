/*
    appellation: mask <module>
    authors: @FL03
*/
//! this module implements various _masks_ often used in neural networks and other machine
//! learning applications to prevent overfitting or to control the flow of information
//! through the network.
#[doc(inline)]
pub use self::prelude::*;

/// An implementation of the dropout regularization technique where randomly selected elements
/// within a tensor/layer are zeroed out during training to prevent overfitting.
mod dropout;

mod prelude {
    #[doc(inline)]
    pub use super::dropout::*;
}
