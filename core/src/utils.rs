/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

// pub(crate) mod checks;
pub(crate) mod math;
pub(crate) mod tensor;

pub(crate) mod prelude {
    // pub use super::checks::*;
    pub use super::math::*;
    pub use super::tensor::*;
}
