extern crate concision_core as cnc;
extern crate concision_models as models;
extern crate concision_neural as neural;

use concision_neural::ModelFeatures;
use concision_neural::model::{Model, StandardModelConfig};
use ndarray::prelude::*;
use simple::SimpleModel;

#[test]
fn test_standard_model_config() {
    // initialize a new model configuration with then given epochs and batch size
    let mut config = StandardModelConfig::new()
        .with_epochs(1000)
        .with_batch_size(32);
    // set various hyperparameters
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // verify the configuration
    assert_eq!(config.batch_size(), 32);
    assert_eq!(config.epochs(), 1000);
    // validate the stored hyperparameters
    assert_eq!(config.learning_rate(), Some(&0.01));
    assert_eq!(config.momentum(), Some(&0.9));
    assert_eq!(config.decay(), Some(&0.0001));
}

#[test]
fn test_simple_model() -> anyhow::Result<()> {
    let mut config = StandardModelConfig::<f64>::new()
        .with_epochs(1000)
        .with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // define the model features
    let features = ModelFeatures::new(3, 9, 9, 1);
    // initialize the model with the given features and configuration
    let model = SimpleModel::<f64>::new(config, features);
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, model.features().input());

    let expected = Array1::from_elem((model.features().output()), 0.5);
    // forward the input through the model
    let y = model.predict(&input)?;
    // verify the output shape
    assert_eq!(y.dim(), (features.output()));
    // compare the results to what we expected
    assert_eq!(y, expected);

    Ok(())
}
