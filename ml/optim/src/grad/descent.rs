/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use crate::neural::layers::linear::LinearLayer;
use crate::neural::prelude::Forward;
use ndarray::prelude::{Array1, Array2};

fn gradient_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    learning_rate: f64,
    epochs: usize,
) {
    let n_samples = x.shape()[0];

    for epoch in 0..epochs {
        let predictions = model.forward(x);
        let errors = &predictions - y;
        let gradient = x.t().dot(&errors) / n_samples as f64;

        model.update_with_gradient(&gradient, learning_rate);

        // let err = mse(&predictions, y).expect("Error calculating MSE");
    }
}
