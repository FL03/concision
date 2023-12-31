/*
    Appellation: cache <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
// use num::{Complex, Float};

pub struct Cache<T = f64, D = Ix2>
where
    D: Dimension,
{
    cache: Array<T, D>,
}
