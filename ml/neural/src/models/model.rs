/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;

pub struct Model<T> {
    pub bias: T,
    pub weights: Array1<T>,
}
