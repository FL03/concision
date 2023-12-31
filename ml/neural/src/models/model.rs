/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{ModelConfig, ModelParams};
use crate::prelude::{Forward, Gradient, LayerParams, Weighted};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};
use std::ops;

#[derive(Clone, Debug)]
pub struct Model<T = f64>
where
    T: Float,
{
    config: ModelConfig,
    params: ModelParams<T>,
}

impl<T> Model<T>
where
    T: Float,
{
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            params: ModelParams::new(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ModelConfig {
        &mut self.config
    }

    pub fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }

    pub fn with_params(mut self, params: ModelParams<T>) -> Self {
        self.params = params;
        self
    }
}

impl<T> Model<T>
where
    T: NdFloat + Signed,
{
    pub fn gradient(
        &mut self,
        data: &Array2<T>,
        targets: &Array2<T>,
        gamma: T,
        grad: impl Gradient<T>,
    ) -> anyhow::Result<f64> {
        // the number of layers in the model
        let depth = self.params().len();
        // the gradients for each layer
        let mut grads = Vec::with_capacity(self.params().len());
        // a store for the predictions of each layer
        let mut store = vec![data.clone()];
        // compute the predictions for each layer
        for layer in self.clone().into_iter() {
            let pred = layer.forward(&store.last().unwrap());
            store.push(pred);
        }
        // compute the error for the last layer
        let error = store.last().unwrap() - targets;
        // compute the error gradient for the last layer
        let dz = &error * grad.gradient(&error);
        // push the error gradient for the last layer
        grads.push(dz.clone());

        for i in (1..depth).rev() {
            // get the weights for the current layer
            let wt = self.params[i].weights().t();
            // compute the delta for the current layer w.r.t. the previous layer
            let dw = grads.last().unwrap().dot(&wt);
            // compute the gradient w.r.t. the current layer's predictions
            let dp = grad.gradient(&store[i]);
            // compute the gradient for the current layer
            let gradient = dw * &dp;
            grads.push(gradient);
        }
        // reverse the gradients so that they are in the correct order
        grads.reverse();
        // update the parameters for each layer
        for i in 0..depth {
            let gradient = &store[i].t().dot(&grads[i]);
            self.params[i]
                .weights_mut()
                .scaled_add(-gamma, &gradient.t());
        }
        let loss = self.forward(data).mean_sq_err(targets)?;
        Ok(loss)
    }
}

impl<T> Model<T>
where
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

// impl<T> FromIterator<LayerShape> for Model<T>
// where
//     T: Float,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = LayerShape>,
//     {
//         let params = ModelParam::from_iter(iter);
//         Self {

//         }
//     }
// }

// impl<T> FromIterator<LayerParams<T>> for Model<T>
// where
//     T: Float,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = LayerParams<T>>,
//     {
//         Self {
//             children: iter.into_iter().collect(),
//         }
//     }
// }
impl<T> IntoIterator for Model<T>
where
    T: Float,
{
    type Item = LayerParams<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<T, D> Forward<Array<T, D>> for Model<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: Dot<Array2<T>, Output = Array<T, D>> + ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn forward(&self, input: &Array<T, D>) -> Self::Output {
        let mut store = vec![input.clone()];
        for layer in self.clone().into_iter() {
            let pred = layer.forward(&store.last().unwrap());
            store.push(pred);
        }
        store.last().unwrap().clone()
    }
}
