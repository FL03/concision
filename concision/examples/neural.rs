extern crate concision as cnc;

use cnc::activate::{ReLU, Sigmoid};
use cnc::nn::{Model, ModelFeatures, ModelParams, StandardModelConfig};
use cnc::{Backward, Forward, Params};
use ndarray::ScalarOperand;
use ndarray::prelude::*;
use num::{Float, FromPrimitive};

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
        model.backward(&training_input, &expected_output, 0.01)?;
    }
    // forward the input through the model
    let output = model.predict(&input)?;
    tracing::info!("output: {:?}", output);

    Ok(())
}

pub struct SimpleModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: ModelParams<T>,
}

impl<T> SimpleModel<T>
where
    T: Float + FromPrimitive,
{
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self {
        let params = ModelParams::zeros(features);
        SimpleModel {
            config,
            features,
            params,
        }
    }

    #[cfg(feature = "rand")]
    pub fn init(self) -> Self
    where
        cnc::init::rand_distr::StandardNormal: cnc::init::rand_distr::Distribution<T>,
    {
        let params = ModelParams::glorot_normal(self.features);
        SimpleModel { params, ..self }
    }

    pub fn train(&mut self, input: &Array2<T>, target: &Array2<T>) -> cnc::Result<T>
    where
        Self: Backward<Array2<T>, Elem = T, Output = T>,
        T: num::traits::NumAssign,
    {
        let learning_rate = self
            .config
            .learning_rate()
            .copied()
            .unwrap_or(T::from_f32(0.01).unwrap());
        let mut loss = T::zero();
        for _epoch in 0..self.config().epochs() {
            loss += self.backward(input, target, learning_rate)?;
        }
        Ok(loss)
    }
}

impl<T> Backward<Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand + core::ops::AddAssign + core::fmt::Debug,
    Params<T>: Backward<Array1<T>, Elem = T, Output = T> + Forward<Array1<T>, Output = Array1<T>>,
{
    type Elem = T;
    type Output = T;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip(self, input, delta, gamma), target = "model")
    )]
    fn backward(
        &mut self,
        input: &Array1<T>,
        delta: &Array1<T>,
        gamma: T,
    ) -> cnc::Result<Self::Output> {
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
        for (layer, (_i, prev)) in self
            .params_mut()
            .hidden_mut()
            .iter_mut()
            .rev()
            .zip(history.iter().skip(1).rev().enumerate())
        {
            let grad = prev.sigmoid_derivative();
            // propagate the gradient through the layer
            loss += layer.backward(prev, &grad, gamma)?;
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

impl<T> Backward<Array2<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand + core::ops::AddAssign + core::fmt::Debug,
    Params<T>: Backward<Array1<T>, Array1<T>, Elem = T, Output = T>
        + Forward<Array1<T>, Output = Array1<T>>,
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
            let input_row = input.row(i).to_owned();
            let delta_row = delta.row(i).to_owned();
            loss += Backward::<Array1<T>>::backward(self, &input_row, &delta_row, gamma)?;
        }

        Ok(loss)
    }
}

impl<T> Forward<Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand,
    Params<T>: Forward<Array1<T>, Output = Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, input: &Array1<T>) -> cnc::Result<Self::Output>
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
