/*
    appellation: impls <module>
    authors: @FL03
*/
#[doc(inline)]
pub use self::prelude::*;

mod config;
mod model;

mod prelude {
    pub use super::config::*;
    pub use super::model::*;
}
