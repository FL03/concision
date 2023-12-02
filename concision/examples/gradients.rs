use concision::neural::prelude::{Layer, Sigmoid};
use concision::neural::models::ModelParams;
use concision::optim::grad::{gradient, Grad};
use concision::prelude::{linarr, Features, Forward, LayerShape};

use ndarray::prelude::{Array1, Ix2};

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 8);
    let outputs = 4;

    let features = LayerShape::new(inputs, outputs);

    let (epochs, gamma) = (1000, 0.005);

    sample_gradient(epochs, features, gamma, samples)?;

    // sample_model(epochs, features, gamma, samples)?;

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

pub fn sample_model(
    epochs: usize,
    features: LayerShape,
    gamma: f64,
    samples: usize,
) -> anyhow::Result<()> {
    // Generate some example data
    let x = linarr::<f64, Ix2>((samples, features.inputs()))?;
    let y = linarr::<f64, Ix2>((samples, features.outputs()))?;

    let mut shapes = vec![features,];
    shapes.extend((0..3).map(|_| LayerShape::new(features.outputs(), features.outputs())));

    let mut model = ModelParams::<f64>::from_iter(shapes);
    let mut opt = Grad::new(gamma, model.clone(), Sigmoid);

    // println!(
    //     "Targets (dim):\t{:?}\nPredictions:\n\n{:?}\n",
    //     &y.shape(),
    //     model.forward(&x)
    // );

    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = opt.step(&x, &y,)?;
        // let cost = model.grad(gamma, &x, &y);
        losses[e] = cost;
    }
    // model = opt.model().clone();
    // println!("Losses:\n\n{:?}\n", &losses);
    // println!("Trained:\n\n{:?}", model.forward(&x));
    Ok(())
}
