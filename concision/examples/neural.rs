extern crate concision as cnc;

use cnc::activate::{ReLU, Sigmoid};
use cnc::nn::{Model, ModelConfig, ModelFeatures, ModelParams};
use ndarray::{Array1, ScalarOperand};
use num::Float;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();
    tracing::info!("Setting up the model...");
    // define the models features
    let features = ModelFeatures::new(3, 9, 9, 2);
    // initialize the models configuration
    let mut config = ModelConfig::new().with_epochs(1000).with_batch_size(32);
    config.set_learning_rate(0.01);
    config.set_momentum(0.9);
    config.set_decay(0.0001);
    // initialize the model
    let model = SimpleModel::<f64>::new(config, features);
    // verify the model configuration
    assert_eq!(model.config().learning_rate(), Some(&0.01));
    assert_eq!(model.config().momentum(), Some(&0.9));
    assert_eq!(model.config().decay(), Some(&0.0001));
    assert_eq!(model.config().epochs(), 1000);
    assert_eq!(model.config().batch_size(), 32);
    // initialize some input data
    let input = Array1::linspace(1.0, 9.0, model.features().input());
    // propagate the input through the model
    let output = model.forward(&input)?;
    tracing::info!("output: {:?}", output);
    // verify the output shape
    assert_eq!(output.dim(), (model.features().output()));
    Ok(())
}

pub struct SimpleModel<T = f64>
where
    T: Float,
{
    pub config: ModelConfig<T>,
    pub features: ModelFeatures,
    pub params: ModelParams<T>,
}

impl<T> SimpleModel<T>
where
    T: Float,
{
    pub fn new(config: ModelConfig<T>, features: ModelFeatures) -> Self {
        let params = ModelParams::zeros(features);
        SimpleModel {
            config,
            features,
            params,
        }
    }
}

impl<T> Model<T> for SimpleModel<T>
where
    T: Float,
{
    fn config(&self) -> &ModelConfig<T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ModelConfig<T> {
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

impl<T> cnc::Forward<Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand,
    cnc::Params<T>: cnc::Forward<Array1<T>, Output = Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, input: &Array1<T>) -> Result<Self::Output, cnc::Error>
    where
        T: Clone,
    {
        let mut output = self.params().input().forward(input)?.relu();

        for layer in self.params().hidden() {
            output = layer.forward(&output)?.sigmoid();
        }

        let res = self.params().output().forward(&output)?;
        Ok(res.relu())
    }
}
