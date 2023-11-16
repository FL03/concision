use concision_neural::prelude::Features;
use concision_optim::prelude::Norm;
use ndarray::prelude::Array;

fn main() -> anyhow::Result<()> {
    let (samples, inputs) = (20, 5);
    let outputs = 1;

    let features = Features::new(inputs, outputs);

    // basic_descent(epochs, features, gamma)?;

    sample_norm(features, samples)?;

    Ok(())
}

pub fn sample_norm(features: Features, samples: usize) -> anyhow::Result<()> {
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

    Ok(())
}
