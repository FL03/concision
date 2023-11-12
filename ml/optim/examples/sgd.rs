use concision_neural::layers::linear::LinearLayer;
use concision_optim::grad::sgd::StochasticGradientDescent;
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 10);
    let shape = (samples, inputs);

    let n = samples * inputs;

    let (batch_size, epochs, gamma) = (20, 4, 0.01);
    // Generate some example data
    let base = Array::linspace(1., n as f64, n);
    let x = base.clone().into_shape(shape).unwrap();
    let y = base.clone().into_shape(n).unwrap() + 1.0;

    let model = LinearLayer::<f64>::new_biased(inputs, 8);

    let mut sgd = StochasticGradientDescent::new(batch_size, epochs, gamma, model);
    let losses = sgd.sgd(&x, &y);
    println!("Losses {:?}", losses);
    Ok(())
}
