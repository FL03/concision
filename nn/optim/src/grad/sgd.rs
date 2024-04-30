/*
    Appellation: sgd <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Stochastic Gradient Descent (SGD)
//!
//!

use neural::prelude::{Activate, Features, Forward, Layer};
// use crate::prelude::ObjectiveFn;
use ndarray::{s, Array1, Array2, Axis, Ix2, NdFloat};
use ndarray_stats::DeviationExt;
use num::{Float, FromPrimitive, Signed};
use rand::seq::SliceRandom;

pub fn sgd<A>(
    x: &Array2<f64>,
    y: &Array2<f64>,
    model: &mut Layer<f64, A>,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
) -> anyhow::Result<Array1<f64>>
where
    A: Clone + Activate<f64, Ix2>,
{
    let layer = model.clone();
    let features = layer.features();
    let (samples, _inputs) = x.dim();
    let mut indices: Vec<usize> = (0..samples).collect();
    let mut losses = Array1::<f64>::zeros(epochs);

    for epoch in 0..epochs {
        indices.shuffle(&mut rand::thread_rng());
        let pos = &indices[..batch_size];

        let xs = x.select(Axis(0), pos);
        let ys = y.select(Axis(0), pos);

        let pred = model.forward(&xs);
        let _error = &pred - &ys;

        for batch in (0..samples).step_by(batch_size) {
            let mut gradient = Array2::zeros((features.outputs(), features.inputs()));

            for i in batch..(batch + batch_size).min(samples) {
                let idx = indices[i];

                let input = x
                    .slice(s![idx, ..])
                    .to_shape((1, features.inputs()))?
                    .to_owned(); // (1, inputs)
                let prediction = model.forward(&input); // (1, outputs)

                let inner = y - &prediction;
                let partial_w = (-2.0 / batch_size as f64) * input.dot(&inner);
                let _partial_b = (-2.0 / batch_size as f64) * inner;
                gradient -= partial_w.sum();
                // let mut weights = model.weights_mut().slice_mut(s![])
                // model.set_weights(weights)

                let cost = y.mean_sq_err(&prediction)?;
                losses[epoch] += cost;
                // let error = &prediction - y[idx];
                println!("Cost:\t{:?}", &cost);
                // gradient += &(input * cost);
            }
            gradient /= batch_size as f64;
            model
                .params_mut()
                .weights_mut()
                .scaled_add(-learning_rate, &gradient.t());

            println!("Gradient:\n{:?}", &gradient);
        }
        losses /= batch_size as f64;
    }

    Ok(losses)
}

pub fn sgd_step<A>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut Layer<f64, A>,
    _learning_rate: f64,
    batch_size: usize,
) -> anyhow::Result<f64>
where
    A: Clone + Activate<f64, Ix2>,
{
    let layer = model.clone();
    let features = layer.features();
    let (_samples, _inputs) = x.dim();
    let mut indices: Vec<usize> = (0..features.outputs()).collect();
    let losses = 0.0;

    indices.shuffle(&mut rand::thread_rng());
    let pos = &indices[..batch_size];

    let xs = x.select(Axis(0), pos);
    let _ys = y.select(Axis(0), pos);

    let _pred = model.forward(&xs);

    Ok(losses)
}

pub struct Sgd {
    batch_size: usize,
    gamma: f64, // learning rate
    model: Layer,
}

impl Sgd {
    pub fn step(&mut self) -> f64 {
        let loss = 0.0;

        loss
    }
}

impl Iterator for Sgd {
    type Item = Array1<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

pub struct StochasticGradientDescent<T = f64>
where
    T: Float,
{
    batch_size: usize,
    epochs: usize,
    gamma: T, // learning rate
    model: Layer<T>,
}

impl<T> StochasticGradientDescent<T>
where
    T: Float,
{
    pub fn new(batch_size: usize, epochs: usize, gamma: T, model: Layer<T>) -> Self {
        Self {
            batch_size,
            epochs,
            gamma,
            model,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn gamma(&self) -> T {
        self.gamma
    }

    pub fn model(&self) -> &Layer<T> {
        &self.model
    }
}

impl<T> StochasticGradientDescent<T>
where
    T: Default + FromPrimitive + NdFloat + Signed,
{
    pub fn sgd(&mut self, x: &Array2<T>, y: &Array2<T>) -> Array1<T> {
        let (samples, inputs) = x.dim();
        let mut indices: Vec<usize> = (0..samples).collect();
        let mut losses = Array1::<T>::zeros(self.epochs);

        for epoch in 0..self.epochs {
            indices.shuffle(&mut rand::thread_rng());

            for batch_start in (0..samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(samples);
                let mut gradient = Array2::zeros((inputs, self.model().features().outputs()));

                for i in batch_start..batch_end {
                    let idx = indices[i];
                    let input = x
                        .slice(s![idx, ..])
                        .to_shape((1, inputs))
                        .expect("")
                        .to_owned(); // (1, inputs)
                    let prediction = self.model.forward(&input); // (1, outputs)
                    let error = &prediction - y;
                    gradient += &(&input * &error.t()).t();
                }

                gradient /= T::from(self.batch_size).unwrap();
                self.model.update_with_gradient(self.gamma, &gradient);

                println!("Gradient:\n{:?}", &gradient);
                let loss = y.mean_sq_err(&self.model.forward(x)).unwrap();
                println!("Epoch: {:?}\nLoss:\n{:?}", &epoch, &loss);
                losses[epoch] += gradient.mean().unwrap_or_default();
            }
            losses[epoch] /= T::from(self.batch_size).unwrap();
        }
        losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::prelude::{LayerShape, Sigmoid};

    #[test]
    fn test_sgd() {
        let (samples, inputs) = (20, 5);
        let outputs = 4;

        let features = LayerShape::new(inputs, outputs);

        let (_bs, _epochs, _gamma) = (10, 1, 0.01);
        // Generate some example data
        let x = linarr::<f64, Ix2>((samples, inputs)).unwrap();
        let _y = linarr::<f64, Ix2>((samples, outputs)).unwrap();

        let model = Layer::<f64, Sigmoid>::from(features).init(true);

        let _pred = model.forward(&x);

        // let mut sgd = StochasticGradientDescent::new(batch_size, epochs, gamma, model);
        // sgd.sgd(&x, &y);
        // let sgd = sgd(&x, &y, &mut model, epochs, gamma, batch_size).unwrap();
    }
}
