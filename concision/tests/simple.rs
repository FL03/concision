/*
    appellation: simple <test>
    authors: @FL03
*/
use cnc::models::ex::sample::TestModel;
use cnc::{Model, ModelFeatures, StandardModelConfig};
use ndarray::prelude::*;

#[test]
fn test_simple_model() -> anyhow::Result<()> {
    let mut config = StandardModelConfig::new()
        .with_epochs(1000)
        .with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // define the model features
    let features = ModelFeatures::deep(3, 9, 1, 8);
    // initialize the model with the given features and configuration
    let model = TestModel::<f64>::new(config, features);
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, model.layout().input());
    let expected = Array1::from_elem(model.layout().output(), 0.5);

    // forward the input through the model
    let output = model.predict(&input);
    // verify the output shape
    assert_eq!(output.dim(), (features.output()));
    // compare the results to what we expected
    assert_eq!(output, expected);

    Ok(())
}
