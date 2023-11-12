use concision_neural::layers::linear::LinearLayer;
use concision_optim::grad::gradient_descent_step;
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 1;
    let n = samples * inputs;

    let (_epochs, gamma) = (10, 0.5);
    // Generate some example data
    let x = Array::linspace(1., n as f64, n)
        .into_shape((samples, inputs))
        .unwrap();
    let y = Array::linspace(1., n as f64, outputs)
        .into_shape(outputs)
        .unwrap();

    let features = (inputs, outputs).into();
    let mut model = LinearLayer::<f64>::new(features).init_weight();

    for e in 0.._epochs {
        let cost = gradient_descent_step(&x, &y, &mut model, gamma);
        println!("Step ({}) Cost {:?}", e, cost);
    }
    Ok(())
}
