/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Layer;
use crate::prelude::{Activate, LinearActivation};
use ndarray::prelude::Ix2;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Stack<T = f64, A = LinearActivation>
where
    A: Activate<T, Ix2>,
    T: Float,
{
    layers: Vec<Layer<T, A>>,
}

impl<T, A> Stack<T, A>
where
    A: Activate<T, Ix2> + Default,
    T: Float,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
}
