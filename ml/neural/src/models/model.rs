/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ModelParams;
use crate::prelude::Features;

pub struct Model<T = f64> {
    pub features: Features,
    children: Vec<Model<T>>,
    params: ModelParams<T>,
}
