use concision::neural::prelude::{Layer, Sigmoid};
use concision::optim::grad::gradient;
use concision::prelude::{linarr, Features, Forward, LayerShape};

use ndarray::prelude::Array1;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 8);
    let outputs = 4;

    let features = LayerShape::new(inputs, outputs);

    let (epochs, gamma) = (1000, 0.005);

    sample_gradient(epochs, features, gamma, samples)?;

    Ok(())
}

pub fn sample_gradient(
    epochs: usize,
    features: LayerShape,
    gamma: f64,
    samples: usize,
) -> anyhow::Result<()> {
    // Generate some example data
    let x = linarr((samples, features.inputs()))?;
    let y = linarr((samples, features.outputs()))?;

    let mut model = Layer::<f64>::from(features).init(false);
    println!(
        "Targets (dim):\t{:?}\nPredictions:\n\n{:?}\n",
        &y.shape(),
        model.forward(&x)
    );

    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = gradient(gamma, &mut model, &x, &y, Sigmoid);
        // let cost = model.grad(gamma, &x, &y);
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);
    println!("Trained:\n\n{:?}", model.forward(&x));
    Ok(())
}
