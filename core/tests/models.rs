/*
    Appellation: models <module>
    Created At: 2025.12.07:11:02:49
    Contrib: @FL03
*/
use concision_core::ex::sample::TestModel;
use concision_core::{Model, ModelFeatures, StandardModelConfig};
use ndarray::prelude::*;

#[test]
fn test_simple_model() {
    // define the model features
    let features = ModelFeatures::deep(3, 9, 1, 8);
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, features.input());
    // initialize a model configuration
    let mut config = StandardModelConfig::new()
        .with_epochs(1000)
        .with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // define and initialize a new model
    let model = TestModel::<f64>::new(config, features).init();
    // forward the input through the model
    let output = model.predict(&input);
    // verify the shape of the output
    assert_eq! { output.dim(), (features.output()) }
}
