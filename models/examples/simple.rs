use concision::nn::{ModelFeatures, Predict, StandardModelConfig, Train};
use concision_models::simple::SimpleModel;
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();
    tracing::info!("Setting up the model...");
    // define the models features
    let features = ModelFeatures::new(3, 9, 9, 2);
    tracing::debug!("Model Features: {features:?}");
    // initialize the models configuration
    let mut config = StandardModelConfig::new()
        .with_epochs(1000)
        .with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    tracing::debug!("Model Config: {config:?}");
    // initialize the model
    let mut model = SimpleModel::<f64>::new(config, features).init();
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, model.features().input());
    // propagate the input through the model
    let output = model.predict(&input)?;
    tracing::info!("output: {:?}", output);
    // verify the output shape
    assert_eq!(output.dim(), (model.features().output()));
    let training_input =
        Array2::from_shape_vec((1, model.features().input()), input.to_vec()).unwrap();
    let expected_output = Array2::from_elem((1, model.features().output()), 0.235);
    // train the model
    for _ in 0..model.config().epochs() {
        model.train(&training_input, &expected_output)?;
    }
    // forward the input through the model
    let output = model.predict(&input)?;
    tracing::info!("output: {:?}", output);

    Ok(())
}
