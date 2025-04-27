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

impl<T> cnc::Backward<Array1<T>, Array1<T>> for SimpleModel<T>
where
    T: Float + ndarray::ScalarOperand + core::ops::AddAssign,
    Params<T>: cnc::Backward<Array1<T>, Array1<T>, Elem = T, Output = T>
        + cnc::Forward<Array1<T>, Output = Array1<T>>,
{
    type Elem = T;
    type Output = T;

    fn backward(
        &mut self,
        input: &Array1<T>,
        delta: &Array1<T>,
        gamma: T,
    ) -> cnc::Result<Self::Output> {
        use cnc::activate::{ReLU, Sigmoid};
        let mut loss = T::zero();
        let mut history = Vec::new();

        let mut output = self.params().input().forward(input)?.relu();
        history.push(output.clone());
        for layer in self.params().hidden() {
            output = layer.forward(&output)?.sigmoid();
            history.push(output.clone());
        }
        output = self.params().output().forward(&output)?.relu();
        history.push(output.clone());

        // compute the gradient of the output layer;
        let grad = (&output - delta).relu_derivative();
        loss += self
            .params_mut()
            .output_mut()
            .backward(&output, &grad, gamma)?;
        // iterate through all of the outputs from the network,
        // skipping the first result and in reverse
        for (i, h) in history.iter().skip(1).rev().enumerate() {
            let layer = &mut self.params_mut().hidden_mut()[i - 1];
            let grad = h.sigmoid_derivative();
            // propagate the gradient through the layer
            loss += layer.backward(h, &grad, gamma)?;
        }
        // propagate the error to the input layer
        let grad = history[0].relu_derivative();
        loss += self
            .params_mut()
            .input_mut()
            .backward(&history[1], &grad, gamma)?;

        Ok(loss)
    }
}

impl<T> cnc::Backward<Array2<T>, Array2<T>> for SimpleModel<T>
where
    T: Float + ndarray::ScalarOperand + core::ops::AddAssign,
    Params<T>: cnc::Backward<Array1<T>, Array1<T>, Elem = T, Output = T>
        + cnc::Forward<Array1<T>, Output = Array1<T>>,
{
    type Elem = T;
    type Output = T;

    fn backward(
        &mut self,
        input: &Array2<T>,
        delta: &Array2<T>,
        gamma: T,
    ) -> cnc::Result<Self::Output> {
        if input.nrows() != delta.nrows() {
            return Err(
                ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
            );
        }

        let mut loss = T::zero();

        for i in 0..input.nrows() {
            let input_row = input.row(i);
            let delta_row = delta.row(i);
            loss += self.backward(&input_row.to_owned(), &delta_row.to_owned(), gamma)?;
        }

        Ok(loss)
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
fn test_simple_model() -> cnc::Result<()> {
    let mut config = StandardModelConfig::new()
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
    // forward the input through the model
    let output = model.predict(&input)?;
    // verify the output shape
    assert_eq!(output.dim(), (features.output()));
    // compare the results to what we expected
    assert_eq!(output, array![0.0]);

    Ok(())
}
