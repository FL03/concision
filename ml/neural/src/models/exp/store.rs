/*
    Appellation: stack <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{LayerParams, LayerPosition, LayerShape};
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

// use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// use std::ops;

pub struct ModelStore<T = f64> {
    store: HashMap<LayerPosition, LayerParams<T>>,
}

impl<T> ModelStore<T>
where
    T: Clone + Default,
{
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            store: HashMap::with_capacity(cap),
        }
    }

    pub fn build_layers(mut self, shapes: impl IntoIterator<Item = (usize, usize)>) -> Self {
        // let shapes = shapes.into_iter().map(|s| (s.inputs(), s.outputs()));
        let tmp = Vec::from_iter(shapes.into_iter().map(|(i, o)| LayerShape::new(i, o)));
        for (i, features) in tmp.iter().enumerate() {
            let position = if i == 0 {
                LayerPosition::input()
            } else if i == tmp.len() - 1 {
                LayerPosition::output(i)
            } else {
                LayerPosition::hidden(i)
            };
            self.store.insert(position, LayerParams::new(*features));
        }
        self
    }

    pub fn with_shapes<Sh>(shapes: impl IntoIterator<Item = Sh>) -> Self
    where
        Sh: IntoDimension<Dim = Ix2>,
    {
        let tmp = Vec::from_iter(shapes.into_iter().map(IntoDimension::into_dimension));
        let mut store = HashMap::new();
        for (i, (inputs, outputs)) in tmp.iter().map(|s| s.into_pattern()).enumerate() {
            let features = LayerShape::new(inputs, outputs);
            let position = if i == 0 {
                LayerPosition::input()
            } else if i == tmp.len() - 1 {
                LayerPosition::output(i)
            } else {
                LayerPosition::hidden(i)
            };
            store.insert(position, LayerParams::new(features));
        }
        Self { store }
    }
}

impl<T> ModelStore<T>
where
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.store.iter_mut().for_each(|(_, l)| {
            *l = l.clone().init(biased);
        });
        self
    }

    pub fn init_bias(mut self) -> Self {
        self.store
            .iter_mut()
            .for_each(|(_, l)| *l = l.clone().init_bias());
        self
    }

    pub fn init_weight(mut self) -> Self {
        self.store
            .iter_mut()
            .for_each(|(_, l)| *l = l.clone().init_weight());
        self
    }
}

impl<T> IntoIterator for ModelStore<T>
where
    T: Float,
{
    type Item = (LayerPosition, LayerParams<T>);
    type IntoIter = std::collections::hash_map::IntoIter<LayerPosition, LayerParams<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_store() {
        let (inputs, outputs) = (5, 3);

        let shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let params = ModelStore::<f64>::new().build_layers(shapes).init(true);

        // validate the dimensions of the model params
        // assert!(params.validate_shapes());

        for (pos, layer) in params.into_iter() {
            let shape = shapes[pos.index()];
            let features = LayerShape::new(shape.0, shape.1);
            assert_eq!(layer.features(), &features);
        }
    }
}
