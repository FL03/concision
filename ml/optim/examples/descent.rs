use concision_core::prelude::linarr;
use concision_neural::prelude::{Features, Layer, Linear, LinearActivation, Sigmoid};
use concision_neural::prop::Forward;
use concision_optim::prelude::{gradient, gradient_descent, GradientDescent};
use ndarray::prelude::Array1;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 8);
    let outputs = 4;

    let features = Features::new(inputs, outputs);

    let (epochs, gamma) = (100000, 0.005);

    // basic_descent(epochs, features, gamma)?;

    // sample_descent(epochs, features, gamma, samples)?;

    sample_gradient(epochs, features, gamma, samples)?;

    Ok(())
}

pub fn basic_descent(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let mut model = Linear::new(features.inputs()).init_weight();

    println!(
        "{:?}",
        gradient_descent(model.weights_mut(), epochs, gamma, Sigmoid::gradient)
    );
    Ok(())
}

pub fn sample_descent(
    epochs: usize,
    features: Features,
    gamma: f64,
    samples: usize,
) -> anyhow::Result<()> {
    // Generate some example data
    let x = linarr((samples, features.inputs()))?;
    let y = linarr(samples)?;

    let model = Linear::new(features.inputs()).init_weight();
    println!(
        "Targets:\n\n{:?}\nPredictions:\n\n{:?}\n",
        &y,
        model.forward(&x)
    );

    let mut grad = GradientDescent::new(gamma, model);
    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = grad.gradient(&x, &y, Sigmoid::gradient)?;
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);

    println!("Trained:\n\n{:?}", grad.model().forward(&x));
    Ok(())
}

pub fn sample_gradient(
    epochs: usize,
    features: Features,
    gamma: f64,
    samples: usize,
) -> anyhow::Result<()> {
    // Generate some example data
    let x = linarr((samples, features.inputs()))?;
    let y = linarr((samples, features.outputs()))?;

    let mut model = Layer::<f64, LinearActivation>::input(features).init(false);
    println!(
        "Targets:\n\n{:?}\nPredictions:\n\n{:?}\n",
        &y,
        model.forward(&x)
    );

    let mut losses = Array1::zeros(epochs);
    for e in 0..epochs {
        let cost = gradient(gamma, &mut model, &x, &y, Sigmoid::gradient);
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);
    println!("Trained:\n\n{:?}", model.forward(&x));
    Ok(())
}
