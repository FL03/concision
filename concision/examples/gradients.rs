use concision::prelude::{linarr, Features, Forward, LayerShape, ParameterizedExt};

use concision::neural::prelude::{Layer, Neuron, Objective, Sigmoid};
use concision::optim::grad::*;

use ndarray::prelude::Array1;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 8);
    let outputs = 4;

    let features = LayerShape::new(inputs, outputs);

    let (epochs, gamma) = (1000000, 0.05);

    // basic_descent(epochs, features, gamma)?;

    // sample_descent(epochs, features, gamma, samples)?;

    sample_gradient(epochs, features, gamma, samples)?;

    Ok(())
}

pub fn basic_descent(epochs: usize, features: LayerShape, gamma: f64) -> anyhow::Result<()> {
    let mut model = Neuron::<f64, Sigmoid>::new(features.inputs()).init_weight();

    println!(
        "{:?}",
        gradient_descent(model.weights_mut(), epochs, gamma, |w| Sigmoid::new()
            .gradient(w))
    );
    Ok(())
}

pub fn sample_descent(
    epochs: usize,
    features: LayerShape,
    gamma: f64,
    samples: usize,
) -> anyhow::Result<()> {
    // Generate some example data
    let x = linarr((samples, features.inputs()))?;
    let y = linarr(samples)?;

    let model = Neuron::new(features.inputs()).init_weight();
    println!(
        "Targets:\n\n{:?}\nPredictions:\n\n{:?}\n",
        &y,
        model.forward(&x)
    );

    let mut grad = GradientDescent::new(gamma, model);
    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = grad.gradient(&x, &y, |w| Sigmoid::new().gradient(w))?;
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);
    println!("Trained:\n\n{:?}", grad.model().forward(&x));
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

    let mut model = Layer::<f64>::input(features).init(false);
    println!(
        "Targets (dim):\t{:?}\nPredictions:\n\n{:?}\n",
        &y.shape(),
        model.forward(&x)
    );

    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = gradient(gamma, &mut model, &x, &y, |w| Sigmoid::new().gradient(w));
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);
    println!("Trained:\n\n{:?}", model.forward(&x));
    Ok(())
}
