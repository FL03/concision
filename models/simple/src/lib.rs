
extern crate concision_core as cnc;

use concision_core::activate::{ReLU, Sigmoid};
use concision_core::{Backward, Forward, Params};
use concision_neural::model::{Model, ModelParams, StandardModelConfig};
use concision_neural::ModelFeatures;

use ndarray::{Array1, Array2, ScalarOperand, ShapeError};
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

impl<T> Backward<Array1<T>, Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand + core::ops::AddAssign,
    Params<T>: Backward<Array1<T>, Array1<T>, Elem = T, Output = T>
        + Forward<Array1<T>, Output = Array1<T>>,
{
    type Elem = T;
    type Output = T;

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

impl<T> Backward<Array2<T>, Array2<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand + core::ops::AddAssign,
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
                ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
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
impl<T> Forward<Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand,
    Params<T>: Forward<Array1<T>, Output = Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, input: &Array1<T>) -> cnc::Result<Self::Output> {
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
