/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::Activate;
use ndarray::prelude::Array2;
use num::Float;

pub trait Model<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn forward(&self, args: &Array2<T>) -> Array2<T>;
}
