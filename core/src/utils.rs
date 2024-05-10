/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub(crate) mod id;
pub(crate) mod math;
pub(crate) mod tensor;

pub(crate) mod prelude {
    #[allow(unused_imports)]
    #[doc(hidden)]
    pub use super::id::*;
    pub use super::math::*;
    pub use super::tensor::*;
}
