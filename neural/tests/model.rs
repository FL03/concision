/*
    Appellation: model <test>
    Contrib: @FL03
*/
extern crate concision_core as cnc;
extern crate concision_neural as neural;

use cnc::Params;
use ndarray::prelude::*;
use neural::model::{Model, StandardModelConfig};
use neural::{ModelFeatures, ModelParams};
use num_traits::Float;

pub struct SimpleModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: ModelParams<T>,
}

impl<T> SimpleModel<T>
where
    T: Float,
{
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self {
        let params = ModelParams::zeros(features);
        SimpleModel {
            config,
            features,
            params,
        }
    }
}

impl<T> cnc::Forward<Array1<T>> for SimpleModel<T>
where
    T: Float + ndarray::ScalarOperand,
    Params<T>: cnc::Forward<Array1<T>, Output = Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, input: &Array1<T>) -> cnc::Result<Self::Output> {
        use cnc::activate::{ReLU, Sigmoid};
        let mut output = self.params().input().forward(input)?.relu();

        for layer in self.params().hidden() {
            output = layer.forward(&output)?.sigmoid();
        }

        self.params().output().forward(&output).map(|y| y.relu())
    }
}

impl<T> Model<T> for SimpleModel<T> {
    type Config = StandardModelConfig<T>;

    fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }

    fn features(&self) -> ModelFeatures {
        self.features
    }

    fn params(&self) -> &ModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ModelParams<T> {
        &mut self.params
    }
}

#[test]
fn test_simple_model() -> cnc::Result<()> {
    let mut config = StandardModelConfig::new()
        .with_epochs(1000)
        .with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // define the model features
    let features = ModelFeatures::new(3, 9, 9, 1);
    // initialize the model
    let model = SimpleModel::<f64>::new(config, features);
    // validate the models configuration
    assert_eq!(model.config().learning_rate(), Some(&0.01));
    assert_eq!(model.config().momentum(), Some(&0.9));
    assert_eq!(model.config().decay(), Some(&0.0001));
    assert_eq!(model.config().epochs(), 1000);
    assert_eq!(model.config().batch_size(), 32);
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, model.features().input());
    // forward the input through the model
    let output = model.predict(&input)?;
    // verify the output shape
    assert_eq!(output.dim(), (features.output()));
    // compare the results to what we expected
    assert_eq!(output, array![0.0]);

    Ok(())
}
