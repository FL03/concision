/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, types::*};

mod constants {
    pub const DEFAULT_BUFFER: usize = 1024;
}

mod types {
    use crate::prelude::{Forward, MlError};
    use ndarray::prelude::{Array, Ix2};

    pub type BoxedFunction<T> = Box<dyn Fn(T) -> T>;
    ///
    pub type ForwardDyn<T = f64, D = Ix2> = Box<dyn Forward<Array<T, D>, Output = Array<T, D>>>;
    ///
    pub type LayerBias<T = f64> = ndarray::Array1<T>;

    pub type LayerWeight<T = f64> = ndarray::Array2<T>;

    pub type Result<T = ()> = core::result::Result<T, MlError>;
}
