/*
    Appellation: optimizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::Params;
use ndarray::prelude::Dimension;

pub struct Optimizer {
    params: Vec<Box<dyn Params>>,
}
