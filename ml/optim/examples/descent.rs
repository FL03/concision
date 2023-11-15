use concision_neural::layers::linear;
use concision_neural::prelude::{Features,  Linear, Neuron};
use concision_neural::prop::Forward;
use concision_optim::prelude::{
    gradient_descent, gradient_descent_node, gradient_descent_step, GradientDescent,
};
use ndarray::prelude::{Array, s};

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 1;

    let features = Features::new(inputs, outputs);

    let n = samples * inputs;

    let (epochs, gamma) = (10, 0.5);

    // basic_descent(epochs, features, gamma)?;

    sample_descent(epochs, features, gamma)?;
    // sample_node(epochs, features.outputs(), gamma)?;
    // sample_steps(epochs, features.outputs(), gamma)?;

    Ok(())
}

pub fn basic_descent(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let (samples, inputs) = (20, features.inputs());
    let n = samples * inputs;

    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
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
        "Initial Prediction: {:?}", &y - model.forward(&x)
    );

    let mut grad = GradientDescent::new(gamma, model);

    for e in 0..epochs {
        let cost = grad.descent(&x, &y);
        println!("Step ({}) Cost {:?}", e, cost);
    }
    println!("Model:\n\nWeights:\n{:?}", grad.model().weights());
    Ok(())
}

pub fn sample_node(epochs: usize, features: usize, gamma: f64) -> anyhow::Result<()> {
    let mut node = Neuron::new(features).init_weights();
    let (samples, inputs) = (20, features);
    let n = samples * inputs;

    // Generate some example data
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., n as f64, samples)
        .into_shape(samples)
        .unwrap();

    for e in 0..epochs {
        let cost = gradient_descent_node(gamma, &mut node, &x, &y);
        println!("Step ({}) Cost {:?}", e, cost);
    }
    Ok(())
}

// pub fn sample_descent(gamma: f64, inputs: usize, outputs: usize, )

pub fn sample_steps(epochs: usize, features: usize, gamma: f64) -> anyhow::Result<()> {
    let mut model = linear::Linear::new(features);
    let (samples, inputs) = (20, features);
    let n = samples * inputs;

    // Generate some example data
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., n as f64, samples)
        .into_shape(samples)
        .unwrap();

    for e in 0..epochs {
        let cost = gradient_descent_step(&x, &y, &mut model, gamma);
        println!("Step ({}) Cost {:?}", e, cost);
    }
    Ok(())
}
