use concision_neural::prelude::{Features, Layer, Sigmoid};
use concision_optim::grad::sgd::sgd;
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 10);
    let outputs = 5;

    let features = Features::new(inputs, outputs);

    let n = samples * inputs;

    let (batch_size, epochs, gamma) = (20, 4, 0.01);
    // Generate some example data
    let base = Array::linspace(1., n as f64, n);
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., n as f64, outputs)
        .into_shape(outputs)
        .unwrap()
        + 1.0;

    let mut model = Layer::<f64, Sigmoid>::new_input(features);

    let cost = sgd(&x, &y, &mut model, epochs, gamma, batch_size).unwrap();
    println!("Losses {:?}", cost);
    Ok(())
}
