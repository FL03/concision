/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub(crate) mod checks;
pub(crate) mod tensor;

#[allow(unused_imports)]
#[doc(hidden)]
pub(crate) mod prelude {
    pub use super::checks::*;
    pub use super::tensor::*;
}
