/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::layers::linear::{LinearLayer, LinearRegression};
use crate::neural::prelude::{mse, Forward, Neuron};
use ndarray::prelude::{Array1, Array2};
use ndarray_stats::DeviationExt;

pub fn cost(target: &Array1<f64>, prediction: &Array1<f64>) -> f64 {
    mse(prediction, target).unwrap_or_default()
}

pub fn gradient_descent(
    data: &mut Array1<f64>,
    epochs: usize,
    gamma: f64,
    obj: impl Fn(&Array1<f64>) -> Array1<f64>,
) -> Array1<f64> {
    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let gradient = obj(data);
        for (xs, g) in data.iter_mut().zip(gradient.iter()) {
            *xs -= gamma * g;
            losses[e] += *xs;
        }
        losses[e] /= data.len() as f64;
    }
    losses
}

pub fn gradient_descent_node(
    gamma: f64,
    node: &mut Neuron,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> f64 {
    let pred = node.forward(x);
    let error = pred.clone() - y;
    let scale = -2.0 / x.len() as f64;
    let wg = scale * x.t().dot(&error);
    node.apply_weight_gradient(gamma, &wg);
    node.forward(x).mean_sq_err(y).unwrap_or_default()
}

pub fn gradient_descent_step(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearRegression,
    learning_rate: f64,
) -> f64 {
    let (samples, _inputs) = x.dim();
    // forward the provided data to the model
    let predictions = model.predict(x); // fully connected
    println!("Predictions (dim): {:?}", &predictions.shape());
    // calculate the error
    let errors = &predictions - y;
    println!("Errors (dim): {:?}", &errors.shape());
    // calculate a scaling factor
    let scale = -2.0 / samples as f64;
    let _bias = scale * &errors;
    // calculate the gradient of the weights
    let weights = scale * x.t().dot(&errors);
    // let wg = (scale * x.t().dot(&errors)).sum_axis(Axis(1));

    println!("Weights (dim): {:?}", &weights.shape());
    model.update_with_gradient(&weights.t().to_owned(), learning_rate);
    // model.apply_gradient(&wg, learning_rate);
    predictions.mean_sq_err(y).unwrap_or_default()
}

pub struct GradientDescent {
    pub gamma: f64,
    model: LinearLayer,
}

impl GradientDescent {
    pub fn new(gamma: f64, model: LinearLayer) -> Self {
        Self { gamma, model }
    }

    pub fn gradient(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array2<f64> {
        let (samples, _inputs) = x.dim();

        let predictions = x.dot(&self.model.weights().t()) + self.model.bias(); // fully connected
        let errors = y - &predictions;
        let scale = -2.0 / samples as f64;
        scale * x.t().dot(&errors)
    }

    pub fn step(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let (samples, _inputs) = x.dim();

        let predictions = x.dot(&self.model.weights().t()) + self.model.bias(); // fully connected
        let errors = y - &predictions;
        let scale = -2.0 / samples as f64;
        let _bias = scale * &errors;
        let weights = scale * x.t().dot(&errors);
        self.model
            .update_with_gradient(&weights.t().to_owned(), self.gamma);

        mse(&self.model.forward(x), y).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::Array;

    #[test]
    fn test_descent() {
        let (samples, inputs) = (20, 5);
        let outputs = 3;
        let n = samples * inputs;

        let (_epochs, gamma) = (1, 0.01);
        // Generate some example data
        let x = Array::linspace(1., n as f64, n)
            .into_shape((samples, inputs))
            .unwrap();
        let y = Array::linspace(1., n as f64, outputs)
            .into_shape(outputs)
            .unwrap();

        let model = LinearLayer::<f64>::new_biased(inputs, outputs);

        let mut grad = GradientDescent::new(gamma, model);

        let _s = grad.step(&x, &y);

        // sgd(&x, &y, &mut model, learning_rate, epochs, batch_size);
    }

    #[test]
    fn test_gradient_descent() {
        let (samples, inputs) = (20, 5);
        let outputs = 5;
        let n = samples * inputs;

        let (_epochs, gamma) = (1, 0.01);
        // Generate some example data
        let x = Array::linspace(1., n as f64, n)
            .into_shape((samples, inputs))
            .unwrap();
        let y = Array::linspace(1., n as f64, samples)
            .into_shape(samples)
            .unwrap();

        // let mut model = LinearLayer::<f64>::new_biased(inputs, outputs);
        let mut model = LinearRegression::new(outputs);

        let _grad = gradient_descent_step(&x, &y, &mut model, gamma);

        // sgd(&x, &y, &mut model, learning_rate, epochs, batch_size);
    }
}
