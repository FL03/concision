/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::Node;

use num::Float;
use petgraph::prelude::{Directed, Graph};

pub enum Edges {
    Biased { bias: f64, weights: Vec<f64> },

    Unbiased { weights: Vec<f64> },

    Empty,
}

pub struct GraphModel<T = f64, K = Directed>
where
    T: Float,
{
    graph: Graph<T, Node<T>, K>,
}
