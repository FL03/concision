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
use ndarray::prelude::{s, Array1, Array2};
use ndarray::ScalarOperand;
use num::{Float, FromPrimitive};
use rand::seq::SliceRandom;
use std::ops;

pub fn sgd(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) -> Array1<f64> {
    let (samples, _inputs) = x.dim();
    let mut indices: Vec<usize> = (0..samples).collect();
    let mut losses = Array1::<f64>::zeros(epochs);

    for epoch in 0..epochs {
        indices.shuffle(&mut rand::thread_rng());
        for batch_start in (0..samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples);
            let mut gradient = Array2::zeros(x.dim());

            for i in batch_start..batch_end {
                let idx = indices[i];
                let input = x.slice(s![idx, ..]).to_owned();
                let prediction = model.forward(&input);
                let error = prediction - y[idx];
                gradient.slice_mut(s![idx, ..]).assign(&(input * error));
            }

            gradient /= batch_size as f64;
            model.update_with_gradient(&gradient, learning_rate);
        }
        let loss = mse(&model.forward(x), y).unwrap();
        losses[epoch] = loss;

        println!("Epoch {}: Loss = {}", epoch, loss);
    }
    losses
}

pub trait Objective<T> {
    type Model;

    fn objective(&self, x: &Array2<T>, y: &Array1<T>) -> Array1<T>;
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
    T: Default + Float + FromPrimitive + ScalarOperand + std::fmt::Debug + ops::AddAssign + ops::DivAssign,
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
                    let input = x.slice(s![idx, ..]).to_shape((1, inputs)).expect("").to_owned(); // (1, inputs)
                    let prediction = self.model.forward(&input); // (1, outputs)
                    let error = &prediction - y[idx];
                    gradient += &(&input * &error.t()).t();
                    
                }

                gradient /= T::from(self.batch_size).unwrap();
                self.model.update_with_gradient(&gradient.t().to_owned(), self.gamma);

                println!("Gradient:\n{:?}", &gradient);
                // let loss = mse(&self.model.forward(x), y).unwrap();
                // println!("Epoch: {:?}\nLoss:\n{:?}", &epoch, &loss);
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
        let y = Array1::<f64>::uniform(0, 100);

        let model = LinearLayer::<f64>::new(inputs, 5);

        let mut sgd = StochasticGradientDescent::new(batch_size, epochs, gamma, model);
        sgd.sgd(&x, &y);

        // sgd(&x, &y, &mut model, learning_rate, epochs, batch_size);
    }
}
