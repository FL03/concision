/*
    Appellation: grad <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::layers::linear::LinearLayer;
use crate::neural::prelude::{mse, Forward};
use ndarray::prelude::{Array1, Array2};

pub fn cost(target: &Array1<f64>, prediction: &Array1<f64>) -> f64 {
    mse(prediction, target).unwrap_or_default()
}

pub fn gradient_descent_step(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &mut LinearLayer,
    learning_rate: f64,
) -> f64 {
    let (_samples, _inputs) = x.dim();

    let predictions = model.forward(x); // fully connected
    let errors = y - &predictions;
    let scale = -1.0 / x.len() as f64;
    let _bias = scale * &errors;
    let weights = scale * x.t().dot(&errors);
    // let wg = (scale * x.t().dot(&errors)).sum_axis(Axis(1));

    println!("Weights: {:?}", &weights);
    model.update_with_gradient(&weights.t().to_owned(), learning_rate);
    // model.apply_gradient(&wg, learning_rate);

    mse(&predictions, y).unwrap_or_default()
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
    fn test_gradient_descent() {
        let (samples, inputs) = (20, 5);
        let outputs = 8;
        let n = samples * inputs;

        let (_epochs, gamma) = (1, 0.01);
        // Generate some example data
        let x = Array::linspace(1., n as f64, n)
            .into_shape((samples, inputs))
            .unwrap();
        let y = Array::linspace(1., n as f64, outputs)
            .into_shape(outputs)
            .unwrap();

        let mut model = LinearLayer::<f64>::new_biased(inputs, outputs);

        let _grad = gradient_descent_step(&x, &y, &mut model, gamma);

        // sgd(&x, &y, &mut model, learning_rate, epochs, batch_size);
    }
}
