use concision_neural::prelude::{Features, Linear};
use concision_neural::prop::Forward;
use concision_optim::prelude::{gradient_descent, GradientDescent};
use ndarray::prelude::{Array, Array1};

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 1;

    let features = Features::new(inputs, outputs);

    let _n = samples * inputs;

    let (epochs, gamma) = (500, 0.01);

    // basic_descent(epochs, features, gamma)?;

    sample_descent(epochs, features, gamma)?;

    Ok(())
}

pub fn basic_descent(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let mut model = Linear::new(features.inputs()).init_weight();

    println!(
        "{:?}",
        gradient_descent(model.weights_mut(), epochs, gamma, |a| a.clone())
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
        let cost = grad.logit(&x, &y)?;
        losses[e] = cost;
    }
    println!("Losses:\n\n{:?}\n", &losses);

    println!("Trained:\n\n{:?}", grad.model().forward(&x));
    Ok(())
}
