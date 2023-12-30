/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{LayerParams, LayerShape};
use crate::func::activate::{Activate, Gradient, Linear};
use crate::prelude::{Features, Forward, Node, Parameterized, Params, Perceptron};
use ndarray::prelude::{Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Layer<T = f64, A = Linear>
where
    A: Activate<T>,
    T: Float,
{
    activator: A,
    features: LayerShape,
    name: String,
    params: LayerParams<T>,
}

impl<T, A> Layer<T, A>
where
    A: Default + Activate<T>,
    T: Float,
{
    pub fn from_features(inputs: usize, outputs: usize) -> Self {
        let features = LayerShape::new(inputs, outputs);
        Self {
            activator: A::default(),
            features,
            name: String::new(),
            params: LayerParams::new(features),
        }
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: Float,
{
    pub fn new(activator: A, features: LayerShape, name: impl ToString) -> Self {
        Self {
            activator,
            features,
            name: name.to_string(),
            params: LayerParams::new(features),
        }
    }

    pub fn activator(&self) -> &A {
        &self.activator
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: impl ToString) {
        self.name = name.to_string();
    }

    pub fn set_node(&mut self, idx: usize, neuron: &Perceptron<T, A>)
    where
        A: Activate<T, Ix1>,
    {
        self.params.set_node(idx, neuron.node().clone());
    }

    pub fn validate_layer(&self, other: &Self, next: bool) -> bool {
        if next {
            return self.features().inputs() == other.features().outputs();
        }
        self.features().outputs() == other.features().inputs()
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T> + Clone + 'static,
    T: Float,
{
    pub fn as_dyn(&self) -> Layer<T, Box<dyn Activate<T>>> {
        Layer {
            activator: Box::new(self.activator.clone()),
            features: self.features.clone(),
            name: self.name.clone(),
            params: self.params.clone(),
        }
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: Float + 'static,
{
    pub fn apply_gradient<F>(&mut self, gamma: T, gradient: F)
    where
        F: Fn(&Array2<T>) -> Array2<T>,
    {
        let grad = gradient(&self.params.weights());
        self.params.weights_mut().scaled_add(-gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array2<T>) {
        self.params.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        self.params().forward(args)
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T> + Gradient<T>,
    T: NdFloat + Signed,
{
    pub fn grad(&mut self, gamma: T, args: &Array2<T>, targets: &Array2<T>) -> T {
        let ns = T::from(args.shape()[0]).unwrap();
        let pred = self.forward(args);

        let scale = T::from(2).unwrap() * ns;

        let errors = &pred - targets;
        let dz = errors * self.activator.gradient(&pred);
        let dw = args.t().dot(&dz) / scale;

        self.params_mut().weights_mut().scaled_add(-gamma, &dw.t());

        let loss = targets
            .mean_sq_err(&pred)
            .expect("Failed to calculate loss");
        T::from(loss).unwrap()
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T, A> Features for Layer<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn inputs(&self) -> usize {
        self.features.inputs()
    }

    fn outputs(&self) -> usize {
        self.features.outputs()
    }
}

// impl<T, D, A> Forward<Array2<T>> for Layer<T, A>
// where
//     A: Activate<T>,
//     D: Dimension,
//     T: NdFloat,
//     Array<T, D>: Dot<Array2<T>, Output = Array<T>>,
// {
//     type Output = Array2<T>;

//     fn forward(&self, args: &Array2<T>) -> Self::Output {
//         self.activator.activate(&self.linear(args))
//     }
// }

impl<T, A> Forward<Array2<T>> for Layer<T, A>
where
    A: Activate<T>,
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.activator.activate(&self.linear(args))
    }
}

impl<T, A> Parameterized<T> for Layer<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Features = LayerShape;
    type Params = LayerParams<T>;

    fn features(&self) -> &LayerShape {
        &self.features
    }

    fn features_mut(&mut self) -> &mut LayerShape {
        &mut self.features
    }

    fn params(&self) -> &LayerParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut LayerParams<T> {
        &mut self.params
    }
}

// impl<T, A> PartialOrd for Layer<T, A>
// where
//     A: Activate<T> + PartialEq,
//     T: Float,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.position.partial_cmp(&other.position)
//     }
// }

// impl<T, A> From<S> for Layer<T, A>
// where
//     A: Activate<T> + Default,
//     S: IntoDimension<Ix2>
//     T: Float,
// {
//     fn from(features: LayerShape) -> Self {
//         Self::new(features, LayerPosition::input())
//     }
// }

impl<T, A> From<LayerShape> for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    fn from(features: LayerShape) -> Self {
        Self {
            activator: A::default(),
            features,
            name: String::new(),
            params: LayerParams::new(features),
        }
    }
}

impl<T, A> IntoIterator for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    type Item = Node<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<T, A> FromIterator<Node<T>> for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let params = LayerParams::from_iter(nodes);
        Self {
            activator: A::default(),
            features: *params.features(),
            name: String::new(),
            params,
        }
    }
}
