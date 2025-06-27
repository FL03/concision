/*
    appellation: impls <module>
    authors: @FL03
*/
#[doc(inline)]
pub use self::prelude::*;

mod model;

mod prelude {
    pub use super::model::*;
}
