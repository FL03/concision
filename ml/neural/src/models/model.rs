/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Activate, Features, LayerParams, Params};
use ndarray::prelude::Ix2;
use num::Float;

pub struct BaseModel<T = f64>
where
    T: Float,
{
    pub features: Features,
    activator: Box<dyn Activate<T, Ix2>>,
    params: Box<dyn Params<T, Ix2>>,
}

pub struct Model<T = f64> {
    pub features: Features,
    children: Vec<Model<T>>,
    layers: usize,

    params: LayerParams<T>,
}
