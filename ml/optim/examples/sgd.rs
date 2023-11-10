use concision_core::prelude::{GenerateRandom, BoxResult};
use concision_neural::layers::linear::LinearLayer;
use concision_optim::grad::sgd::StochasticGradientDescent;
use ndarray::prelude::{Array, Array1};

fn main() -> BoxResult {
    let (samples, inputs) = (20, 5);
    let shape = (samples, inputs);

    let (batch_size, epochs, gamma) = (10, 1, 0.01);
    // Generate some example data
    let x = Array::linspace(1., 100., 100).into_shape(shape).unwrap();
    let y = Array::linspace(1., 100., 100).into_shape(100).unwrap();

    let model = LinearLayer::<f64>::new(inputs, 5);

    let mut sgd = StochasticGradientDescent::new(batch_size, epochs, gamma, model);
    let losses = sgd.sgd(&x, &y);
    println!("Losses {:?}", losses);
    Ok(())
}