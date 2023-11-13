use concision_neural::layers::linear;
use concision_neural::prelude::{Features, Neuron};
use concision_optim::prelude::{
    gradient_descent, gradient_descent_node, gradient_descent_step, GradientDescent,
};
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 1;

    let features = Features::new(inputs, outputs);

    let n = samples * inputs;

    let (epochs, gamma) = (100, 0.5);
    let mut x = Array::linspace(1., n as f64, samples)
        .into_shape(samples)
        .unwrap();
    println!(
        "{:?}",
        gradient_descent(&mut x, epochs, gamma, |a| a.clone())
    );
    // sample_descent(epochs, features, gamma)?;
    // sample_node(epochs, features.outputs(), gamma)?;
    // sample_steps(epochs, features.outputs(), gamma)?;

    Ok(())
}

pub fn sample_descent(epochs: usize, features: Features, gamma: f64) -> anyhow::Result<()> {
    let (samples, inputs) = (20, features.inputs());
    let outputs = features.outputs();
    let n = samples * inputs;

    // Generate some example data
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., n as f64, outputs)
        .into_shape(outputs)
        .unwrap();

    let model = linear::LinearLayer::<f64>::new(features).init_weight();
    let mut grad = GradientDescent::new(gamma, model);

    for e in 0..epochs {
        let cost = grad.step(&x, &y);
        println!("Step ({}) Cost {:?}", e, cost);
    }
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
    let mut model = linear::LinearRegression::new(features);
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
