/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

pub(crate) mod constants {
    pub const DEFAULT_BUFFER: usize = 1024;
}

pub(crate) mod statics {}

pub(crate) mod types {
    use crate::prelude::Forward;
    use ndarray::prelude::{Array, Ix2};

    pub type BoxedFunction<T> = Box<dyn Fn(T) -> T>;
    ///
    pub type ForwardDyn<T = f64, D = Ix2> = Box<dyn Forward<Array<T, D>, Output = Array<T, D>>>;
    ///
    pub type LayerBias<T = f64> = ndarray::Array1<T>;

    pub type LayerWeight<T = f64> = ndarray::Array2<T>;
}
