use concision_core::prelude::linarr;
use concision_neural::prelude::{Features, Layer, Linear, LinearActivation, Sigmoid};
use concision_neural::prop::Forward;
use concision_optim::prelude::{gradient, gradient_descent, GradientDescent};
use ndarray::prelude::{Array, Array1};

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 3;

    let features = Features::new(inputs, outputs);

    let _n = samples * inputs;

    let (epochs, gamma) = (100000, 0.05);

    // basic_descent(epochs, features, gamma)?;

    // sample_descent(epochs, features, gamma)?;

    sample_gradient(epochs, features, gamma)?;

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

pub fn sample_descent(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let (samples, inputs) = (20, features.inputs());
    let n = samples * inputs;

    // Generate some example data
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., samples as f64, samples)
        .into_shape(samples)
        .unwrap();

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

pub fn sample_gradient(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let (samples, inputs) = (20, features.inputs());

    // Generate some example data
    let x = linarr((samples, inputs))?;
    let y = linarr((samples, features.outputs()))?;

    let mut model = Layer::<f64, LinearActivation>::new_input(features).init(false);
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
