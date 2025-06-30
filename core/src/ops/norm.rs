/*
    appellation: norm <module>
    authors: @FL03
*/
//! this module implements various normalization operations for tensors
#[doc(inline)]
pub use self::prelude::*;

mod l_norm;

mod prelude {
    #[doc(inline)]
    pub use super::l_norm::*;
}
