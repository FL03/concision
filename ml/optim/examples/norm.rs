use concision_neural::prelude::LayerShape;
use concision_optim::prelude::Norm;
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 3);
    let outputs = 8;

    let features = LayerShape::new(inputs, outputs);

    // basic_descent(epochs, features, gamma)?;

    sample_norm(features, samples)?;

    Ok(())
}

pub fn sample_norm(features: LayerShape, samples: usize) -> anyhow::Result<()> {
    let n = samples * features.inputs();
    let args = Array::linspace(1., n as f64, n)
        .into_shape((samples, features.inputs()))
        .unwrap();

    println!(
        "Norms:\n\nL0: {:?}\nL1: {:?}\nL2: {:?}\n",
        &args.l0(),
        &args.l1(),
        &args.l2()
    );
    let args = Array::linspace(1., features.inputs() as f64, features.inputs())
        .into_shape(features.inputs())
        .unwrap();
    println!(
        "Norms:\n\nL0: {:?}\nL1: {:?}\nL2: {:?}\n",
        &args.l0(),
        &args.l1(),
        &args.l2()
    );

    Ok(())
}
