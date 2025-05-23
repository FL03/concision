extern crate concision_core as cnc;

use concision_core::activate::{ReLU, Sigmoid};
use concision_core::{Forward, Norm, Params};
use concision_neural::model::{Model, ModelParams, StandardModelConfig};
use concision_neural::{ModelFeatures, NeuralError, Predict, Train};

use ndarray::prelude::*;
use ndarray::{Data, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumAssign};

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

impl<T> Forward<Array1<T>> for SimpleModel<T>
where
    T: Float + ScalarOperand,
    Params<T>: Forward<Array1<T>, Output = Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, input: &Array1<T>) -> cnc::Result<Self::Output> {
        let mut output = self.params().input().forward_then(&input, |y| y.relu())?;

        for layer in self.params().hidden() {
            output = layer.forward_then(&output, |y| y.relu())?;
        }

        self.params()
            .output()
            .forward_then(&output, |y| y.sigmoid())
    }
}

impl<A, S, T> Predict<ArrayBase<S, Ix1>> for SimpleModel<A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    Self: Forward<ArrayBase<S, Ix1>, Output = ArrayBase<T, Ix1>>,
{
    type Confidence = A;
    type Output = <Self as Forward<ArrayBase<S, Ix1>>>::Output;

    fn predict(&self, input: &ArrayBase<S, Ix1>) -> Result<Self::Output, NeuralError> {
        let y = Forward::<ArrayBase<S, Ix1>>::forward(self, input)?;
        Ok(y)
    }

    fn predict_with_confidence(
        &self,
        input: &ArrayBase<S, Ix1>,
    ) -> Result<
        (
            <Self as Predict<ArrayBase<S, Ix1>>>::Output,
            Self::Confidence,
        ),
        NeuralError,
    > {
        // Get the base prediction
        let prediction = Forward::<ArrayBase<S, Ix1>>::forward(self, input)?;
        // Calculate confidence as the inverse of the variance of the output
        // For each sample, compute the variance across the output dimensions
        let variance = prediction.var(A::one());

        // Average variance across the batch
        let avg_variance = variance / A::from_usize(prediction.len()).unwrap();
        // Confidence: inverse of variance (clipped to avoid division by zero)
        let confidence = (A::one() + avg_variance).recip();

        Ok((prediction, confidence))
    }
}

impl<A, S, T> Predict<ArrayBase<S, Ix2>> for SimpleModel<A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    Self: Forward<ArrayBase<S, Ix2>, Output = ArrayBase<T, Ix2>>,
{
    type Confidence = A;
    type Output = <Self as Forward<ArrayBase<S, Ix2>>>::Output;

    fn predict(&self, input: &ArrayBase<S, Ix2>) -> Result<Self::Output, NeuralError> {
        let y = Forward::<ArrayBase<S, Ix2>>::forward(self, input)?;
        Ok(y)
    }

    fn predict_with_confidence(
        &self,
        input: &ArrayBase<S, Ix2>,
    ) -> Result<
        (
            <Self as Predict<ArrayBase<S, Ix2>>>::Output,
            Self::Confidence,
        ),
        NeuralError,
    > {
        // Get the base prediction
        let prediction = Forward::<ArrayBase<S, Ix2>>::forward(self, input)?;
        // Calculate confidence as the inverse of the variance of the output
        // For each sample, compute the variance across the output dimensions
        let batch_size = prediction.shape()[0];
        let mut variance_sum = A::zero();

        for i in 0..batch_size {
            let sample = prediction.row(i);
            // Compute variance
            let variance = sample.var(A::one());
            variance_sum = variance_sum + variance;
        }

        // Average variance across the batch
        let avg_variance = variance_sum / A::from_usize(batch_size).unwrap();
        // Confidence: inverse of variance (clipped to avoid division by zero)
        let confidence = (A::one() + avg_variance).recip();

        Ok((prediction, confidence))
    }
}

impl<A, S, T> Train<ArrayBase<S, Ix1>, ArrayBase<T, Ix1>> for SimpleModel<A>
where
    A: Float + FromPrimitive + NumAssign + ScalarOperand + core::fmt::Debug,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Output = A;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip(self, input, target),
            level = "trace",
            name = "backward",
            target = "model",
        )
    )]
    fn train(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        target: &ArrayBase<T, Ix1>,
    ) -> Result<Self::Output, NeuralError> {
        if input.len() != self.features().input() {
            return Err(NeuralError::InvalidInputShape);
        }
        if target.len() != self.features().output() {
            return Err(NeuralError::InvalidOutputShape);
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

        let mut output = self.params().input().forward(&input)?.relu();
        activations.push(output.to_owned());
        // collect the activations of the hidden
        for layer in self.params().hidden() {
            output = layer.forward(&output)?.relu();
            activations.push(output.to_owned());
        }

        output = self.params().output().forward(&output)?.sigmoid();
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
            .backward(activations.last().unwrap(), &delta, lr)?;

        let num_hidden = self.features().layers();
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
            self.params_mut().hidden_mut()[i].backward(&activations[i + 1], &delta, lr)?;
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
            .backward(&activations[1], &delta, lr)?;

        Ok(loss)
    }
}

impl<A, S, T> Train<ArrayBase<S, Ix2>, ArrayBase<T, Ix2>> for SimpleModel<A>
where
    A: Float + FromPrimitive + NumAssign + ScalarOperand + core::fmt::Debug,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Output = A;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip(self, input, target),
            level = "trace",
            name = "train",
            target = "model",
            fields(input_shape = ?input.shape(), target_shape = ?target.shape())
        )
    )]
    fn train(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        target: &ArrayBase<T, Ix2>,
    ) -> Result<Self::Output, NeuralError> {
        if input.nrows() == 0 || target.nrows() == 0 {
            return Err(NeuralError::InvalidBatchSize);
        }
        if input.ncols() != self.features().input() {
            return Err(NeuralError::InvalidInputShape);
        }
        if target.ncols() != self.features().output() || target.nrows() != input.nrows() {
            return Err(NeuralError::InvalidOutputShape);
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
