/*
    Appellation: sgd <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Stochastic Gradient Descent (SGD)
//!
//!
use crate::layers::linear::LinearLayer;
use crate::nn::loss::mse;
use crate::prelude::Forward;
use ndarray::prelude::{s, Array1, Array2};
use num::Float;
use rand::seq::SliceRandom;

fn sgd(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) -> Array1<f64> {
    let n_samples = x.shape()[0];
    let input_size = x.shape()[1];
    let mut losses = Array1::<f64>::zeros(epochs);

    for epoch in 0..epochs {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let mut gradient = Array2::zeros(x.dim());

            for i in batch_start..batch_end {
                let sample_index = indices[i];
                let input = x.slice(s![sample_index, ..]).to_owned();
                let prediction = model.forward(&input);
                let error = prediction - y[sample_index];
                gradient
                    .slice_mut(s![sample_index, ..])
                    .assign(&(input * error));
            }

            gradient /= batch_size as f64;
            model.update_with_gradient(&gradient, learning_rate);
        }
        let loss = mse(&model.forward(x), y).unwrap();
        losses[epoch] = loss;

        // println!("Epoch {}: Loss = {}", epoch, mse(&model.forward(x), y).unwrap());
    }
    losses
}

pub struct StochasticGradientDescent<T = f64>
where
    T: Float,
{
    batch_size: usize,
    epochs: usize,
    learning_rate: f64,
    model: LinearLayer<T>,
}

impl<T> StochasticGradientDescent<T>
where
    T: Float,
{
    pub fn new(
        model: LinearLayer<T>,
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            batch_size,
            epochs,
            learning_rate,
            model,
        }
    }

    pub fn train(&mut self, x: &Array2<T>, y: &Array1<T>) {}

    pub fn model(&self) -> &LinearLayer<T> {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd() {
        // Generate some example data
        let x = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
        let y = x.dot(&Array1::from_elem(2, 2.0)) + &Array1::from_elem(100, 1.0);

        let mut model = LinearLayer::new(200, 100);
        let learning_rate = 0.01;
        let epochs = 100;
        let batch_size = 10;

        sgd(&x, &y, &mut model, learning_rate, epochs, batch_size);
    }
}
