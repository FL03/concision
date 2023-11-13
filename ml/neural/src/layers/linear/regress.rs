/*
   Appellation: regress <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Axis};
use ndarray::{Dimension, NdFloat};
use ndarray_stats::CorrelationExt;
use num::FromPrimitive;

pub fn covariance<T, D>(ddof: T, x: &Array<T, D>, y: &Array<T, D>) -> anyhow::Result<Array<T, D>>
where
    D: Dimension,
    T: Default + FromPrimitive + NdFloat,
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
{
    let x_mean = x.mean().unwrap_or_default();
    let y_mean = y.mean().unwrap_or_default();
    let xs = x - x_mean;
    let ys = y - y_mean;
    let cov = xs.dot(&ys.t().to_owned());
    let scale = T::one() / (T::from(x.len()).unwrap() - ddof);
    Ok(cov * scale)
}

pub struct LinearRegression {
    bias: f64,
    pub features: usize,
    weights: Array1<f64>,
}

impl LinearRegression {
    pub fn new(features: usize) -> Self {
        Self {
            bias: 0.0,
            features,
            weights: Array1::zeros(features),
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn bias_mut(&mut self) -> &mut f64 {
        &mut self.bias
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn features_mut(&mut self) -> &mut usize {
        &mut self.features
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array1<f64> {
        &mut self.weights
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn set_features(&mut self, features: usize) {
        self.features = features;
    }

    pub fn set_weights(&mut self, weights: Array1<f64>) {
        self.weights = weights;
    }

    pub fn fit(&mut self, data: &Array2<f64>, targets: &Array1<f64>) -> anyhow::Result<()> {
        let _m = data.cov(0.0)? / data.var_axis(Axis(0), 0.0);
        // let covar = covariance(0.0, x, y);
        // self.bias = targets.mean().unwrap_or_default() - m * data.mean().unwrap_or_default();
        // self.weights -= m;
        Ok(())
    }

    pub fn predict(&self, data: &Array2<f64>) -> Array1<f64> {
        data.dot(&self.weights().t()) + self.bias()
    }

    pub fn update_with_gradient(&mut self, gradient: &Array1<f64>, gamma: f64) {
        self.weights = &self.weights - gradient * gamma;
    }
}
