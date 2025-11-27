/*
    appellation: model <test>
    authors: @FL03
*/
use cnc::{DeepModelParams, Model, ModelFeatures, Error, StandardModelConfig};
use cnc::data::Train;
use cnc::{Forward, Norm, Params, ReLU, Sigmoid};

use ndarray::prelude::*;
use ndarray::{Data, ScalarOperand};
use num::traits::{Float, FromPrimitive, NumAssign};

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SimpleModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: DeepModelParams<T>,
}

impl<T> SimpleModel<T>
where
    T: Float,
{
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self {
        let params = DeepModelParams::zeros(features);
        SimpleModel {
            config,
            features,
            params,
        }
    }
}

impl<T> Model<T> for SimpleModel<T> {
    type Config = StandardModelConfig<T>;

    type Layout = ModelFeatures;

    fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }

    fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }

    fn layout(&self) -> ModelFeatures {
        self.features
    }

    fn params(&self) -> &DeepModelParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut DeepModelParams<T> {
        &mut self.params
    }
}

impl<A, S, D> Forward<ArrayBase<S, D>> for SimpleModel<A>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
    Params<A>: Forward<Array<A, D>, Output = Array<A, D>>,
{
    type Output = Array<A, D>;

    fn forward(&self, input: &ArrayBase<S, D>) -> Option<Self::Output> {
        let mut output = self
            .params()
            .input()
            .forward_then(&input.to_owned(), |y| y.relu())?;

        for layer in self.params().hidden() {
            output = layer.forward_then(&output, |y| y.relu())?;
        }

        let y = self
            .params()
            .output()
            .forward_then(&output, |y| y.sigmoid())?;
        Some(y)
    }
}

impl<A, S, T> Train<ArrayBase<S, Ix1>, ArrayBase<T, Ix1>> for SimpleModel<A>
where
    A: Float + FromPrimitive + NumAssign + ScalarOperand + core::fmt::Debug,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Error = Error;
    type Output = A;

    fn train(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        target: &ArrayBase<T, Ix1>,
    ) -> Result<Self::Output, Error> {
        if input.len() != self.layout().input() {
            return Err(Error::InvalidInputShape);
        }
        if target.len() != self.layout().output() {
            return Err(Error::InvalidOutputShape);
        }
        // get the learning rate from the model's configuration
        let lr = self
            .config()
            .learning_rate()
            .copied()
            .unwrap_or(A::from_f32(0.01).unwrap());
        // Normalize the input and target
        let input = input / input.l2_norm();
        let target_norm = target.l2_norm();
        let target = target / target_norm;
        // self.prev_target_norm = Some(target_norm);
        // Forward pass to collect activations
        let mut activations = Vec::new();
        activations.push(input.to_owned());

        let mut output = self
            .params()
            .input()
            .forward(&input)
            .expect("Failed to complete the forward pass for the input layer")
            .relu();
        activations.push(output.to_owned());
        // collect the activations of the hidden
        for layer in self.params().hidden() {
            output = layer
                .forward(&output)
                .expect("failed to complete the forward pass for the hidden layer")
                .relu();
            activations.push(output.to_owned());
        }

        output = self
            .params()
            .output()
            .forward(&output)
            .expect("Output layer failed to forward propagate")
            .sigmoid();
        activations.push(output.to_owned());

        // Calculate output layer error
        let error = &target - &output;
        let loss = error.pow2().mean().unwrap_or(A::zero());
        #[cfg(feature = "tracing")]
        tracing::trace!("Training loss: {loss:?}");
        let mut delta = error * output.sigmoid_derivative();
        delta /= delta.l2_norm(); // Normalize the delta to prevent exploding gradients

        // Update output weights
        self.params_mut()
            .output_mut()
            .backward(activations.last().unwrap(), &delta, lr)
            .expect("Output failed training...");

        let num_hidden = self.layout().layers();
        // Iterate through hidden layers in reverse order
        for i in (0..num_hidden).rev() {
            // Calculate error for this layer
            delta = if i == num_hidden - 1 {
                // use the output activations for the final hidden layer
                self.params().output().weights().dot(&delta) * activations[i + 1].relu_derivative()
            } else {
                // else; backpropagate using the previous hidden layer
                self.params().hidden()[i + 1].weights().t().dot(&delta)
                    * activations[i + 1].relu_derivative()
            };
            // Normalize delta to prevent exploding gradients
            delta /= delta.l2_norm();
            self.params_mut().hidden_mut()[i]
                .backward(&activations[i + 1], &delta, lr)
                .expect("Hidden failed training...");
        }
        /*
            Backpropagate to the input layer
            The delta for the input layer is computed using the weights of the first hidden layer
            and the derivative of the activation function of the first hidden layer.

            (h, h).dot(h) * derivative(h) = dim(h) where h is the number of features within a hidden layer
        */
        delta = self.params().hidden()[0].weights().dot(&delta) * activations[1].relu_derivative();
        delta /= delta.l2_norm(); // Normalize the delta to prevent exploding gradients
        self.params_mut()
            .input_mut()
            .backward(&activations[1], &delta, lr)
            .expect("failed to backpropagate input layer during training...");

        Ok(loss)
    }
}

impl<A, S, T> Train<ArrayBase<S, Ix2>, ArrayBase<T, Ix2>> for SimpleModel<A>
where
    A: Float + FromPrimitive + NumAssign + ScalarOperand + core::fmt::Debug,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Error = Error;
    type Output = A;

    fn train(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        target: &ArrayBase<T, Ix2>,
    ) -> Result<Self::Output, Error> {
        if input.nrows() == 0 || target.nrows() == 0 {
            return Err(Error::InvalidBatchSize);
        }
        if input.ncols() != self.layout().input() {
            return Err(Error::InvalidInputShape);
        }
        if target.ncols() != self.layout().output() || target.nrows() != input.nrows() {
            return Err(Error::InvalidOutputShape);
        }
        let batch_size = input.nrows();
        let mut loss = A::zero();

        for (i, (x, e)) in input.rows().into_iter().zip(target.rows()).enumerate() {
            loss += match Train::<ArrayView1<A>, ArrayView1<A>>::train(self, &x, &e) {
                Ok(l) => l,
                Err(err) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        "Training failed for batch {}/{}: {:?}",
                        i + 1,
                        batch_size,
                        err
                    );
                    return Err(err);
                }
            };
        }

        Ok(loss)
    }
}
