/*
    Appellation: sublayers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::LayerParams;

use ndarray::prelude::Array2;
use num::Float;

pub trait Wrapper<T = f64>
where
    T: Float,
{
    fn apply(&self, data: &Array2<T>) -> Array2<T>;

    fn params(&self) -> &LayerParams<T>;
}
