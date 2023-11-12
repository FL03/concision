/*
    Appellation: sgd <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Stochastic Gradient Descent (SGD)
//!
//!
use crate::neural::layers::linear::LinearLayer;
use crate::neural::prelude::{mse, Forward};
// use crate::prelude::ObjectiveFn;
use ndarray::prelude::{s, Array1, Array2, Axis};
use ndarray::NdFloat;
use num::{Float, FromPrimitive};
use rand::seq::SliceRandom;

pub fn sgd(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
) -> anyhow::Result<Array1<f64>> {
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
        let error = &pred - &ys;
        let grad_w = xs.dot(&error.t()).sum() * (-2.0 / batch_size as f64);
        let grad_b = error.sum() * (-2.0 / batch_size as f64);

        for batch in (0..samples).step_by(batch_size) {
            let mut gradient = Array2::zeros((features.outputs(), features.inputs()));

            for i in batch..(batch + batch_size).min(samples) {
                let idx = indices[i];

                let input = x
                    .slice(s![idx, ..])
                    .to_shape((1, features.inputs()))?
                    .to_owned(); // (1, inputs)
                let prediction = model.forward(&input); // (1, outputs)

                let inner = y[idx] - &prediction;
                let partial_w = (-2.0 / batch_size as f64) * input.dot(&inner);
                let partial_b = (-2.0 / batch_size as f64) * inner;
                gradient -= partial_w.sum();
                // let mut weights = model.weights_mut().slice_mut(s![])
                // model.set_weights(weights)

                let cost = mse(&prediction, y).unwrap();
                losses[epoch] += cost;
                // let error = &prediction - y[idx];
                println!("Cost:\t{:?}", &cost);
                // gradient += &(input * cost);
            }
            gradient /= batch_size as f64;
            model.update_with_gradient(&gradient, learning_rate);

            println!("Gradient:\n{:?}", &gradient);
        }
        losses /= batch_size as f64;
    }

    Ok(losses)
}

pub fn sgd_step(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    learning_rate: f64,
    batch_size: usize,
) -> anyhow::Result<f64> {
    let layer = model.clone();
    let features = layer.features();
    let (samples, _inputs) = x.dim();
    let mut indices: Vec<usize> = (0..features.outputs()).collect();
    let mut losses = 0.0;

    indices.shuffle(&mut rand::thread_rng());
    let pos = &indices[..batch_size];

    let xs = x.select(Axis(0), pos);
    let ys = y.select(Axis(0), pos);

    let pred = model.forward(&xs);

    Ok(losses)
}

pub struct Sgd {
    batch_size: usize,
    gamma: f64, // learning rate
    model: LinearLayer,
}

impl Sgd {
    pub fn step(&mut self) -> f64 {
        let mut loss = 0.0;

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
    model: LinearLayer<T>,
}

impl<T> StochasticGradientDescent<T>
where
    T: Float,
{
    pub fn new(batch_size: usize, epochs: usize, gamma: T, model: LinearLayer<T>) -> Self {
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

    pub fn model(&self) -> &LinearLayer<T> {
        &self.model
    }
}

impl<T> StochasticGradientDescent<T>
where
    T: Default + FromPrimitive + NdFloat,
{
    pub fn sgd(&mut self, x: &Array2<T>, y: &Array1<T>) -> Array1<T> {
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
                self.model
                    .update_with_gradient(&gradient.t().to_owned(), self.gamma);

                println!("Gradient:\n{:?}", &gradient);
                let loss = mse(&self.model.forward(x), y).unwrap();
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
    use crate::core::prelude::GenerateRandom;
    use ndarray::prelude::{Array, Array1};

    #[test]
    fn test_sgd() {
        let (samples, inputs) = (20, 5);
        let shape = (samples, inputs);

        let (batch_size, epochs, gamma) = (10, 1, 0.01);
        // Generate some example data
        let x = Array::linspace(1., 100., 100).into_shape(shape).unwrap();
        let y = Array::linspace(1., 100., 5).into_shape(5).unwrap();

        let model = LinearLayer::<f64>::new_biased(inputs, 5);

        let mut sgd = StochasticGradientDescent::new(batch_size, epochs, gamma, model);
        sgd.sgd(&x, &y);
    }

    #[test]
    fn test_stochastic() {
        let (samples, inputs) = (20, 5);
        let shape = (samples, inputs);

        let (batch_size, epochs, gamma) = (10, 1, 0.01);
        // Generate some example data
        let x = Array::linspace(1., 100., 100).into_shape(shape).unwrap();
        let y = Array1::<f64>::uniform(0, 100);

        let mut model = LinearLayer::<f64>::new_biased(inputs, 5);

        // let grad = sgd(&x, &y, &mut model, epochs, gamma, batch_size);
        // assert!(grad.is_ok());
    }
}
