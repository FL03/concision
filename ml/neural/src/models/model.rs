/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ModelParams;
use crate::prelude::{Activate, Features, Params};
use ndarray::prelude::{Array, Ix2};
use num::Float;

pub struct BaseModel<T = f64>
where
    T: Float,
{
    pub features: Features,
    activator: Box<dyn Activate<T>>,
    params: Box<dyn Params<T, Ix2>>,
}

pub struct Model<T = f64> {
    pub features: Features,
    children: Vec<Model<T>>,
    params: ModelParams<T>,
}
