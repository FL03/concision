use concision_core::prelude::linarr;
use concision_neural::prelude::{Layer, LayerShape, Sigmoid};
use concision_optim::grad::sgd::sgd;
use ndarray::prelude::Array2;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 10);
    let outputs = 5;

    let features = LayerShape::new(inputs, outputs);

    let (batch_size, epochs, gamma) = (20, 4, 0.01);
    // Generate some example data
    let (x, y) = sample_data::<f64>(inputs, outputs, samples)?;

    let mut model = Layer::<f64, Sigmoid>::from(features);

    let cost = sgd(&x, &y, &mut model, epochs, gamma, batch_size).unwrap();
    println!("Losses {:?}", cost);
    Ok(())
}

fn sample_data<T: num::Float>(
    inputs: usize,
    outputs: usize,
    samples: usize,
) -> anyhow::Result<(Array2<T>, Array2<T>)> {
    let x = linarr((samples, inputs)).unwrap(); // (samples, inputs)
    let y = linarr((samples, outputs)).unwrap(); // (samples, outputs)
    Ok((x, y))
}
