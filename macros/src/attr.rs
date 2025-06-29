/*
    appellation: attr <module>
    authors: @FL03
*/
#[doc(inline)]
#[allow(unused_imports)]
pub use self::prelude::*;

mod model;

#[allow(unused_imports)]
mod prelude {
    pub use super::model::*;
}
