/*
    Appellation: simple <module>
    Contrib: @FL03
*/
use cnc::nn::{DeepModelParams, Model, ModelError, ModelFeatures, StandardModelConfig, Train};
use cnc::{Forward, Norm, Params, ReLU, Sigmoid};

use ndarray::prelude::*;
use ndarray::{Data, ScalarOperand};
use num_traits::{Float, FromPrimitive, NumAssign};

#[derive(Clone, Debug)]
pub struct SimpleModel<T = f64> {
    pub config: StandardModelConfig<T>,
    pub features: ModelFeatures,
    pub params: DeepModelParams<T>,
}

impl<T> SimpleModel<T> {
    pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self
    where
        T: Clone + Default,
    {
        let params = DeepModelParams::default(features);
        SimpleModel {
            config,
            features,
            params,
        }
    }
    /// returns a reference to the model configuration
    pub const fn config(&self) -> &StandardModelConfig<T> {
        &self.config
    }
    /// returns a mutable reference to the model configuration
    pub const fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
        &mut self.config
    }
    /// returns the model features
    pub const fn features(&self) -> ModelFeatures {
        self.features
    }
    /// returns a mutable reference to the model features
    pub const fn features_mut(&mut self) -> &mut ModelFeatures {
        &mut self.features
    }
    /// returns a reference to the model parameters
    pub const fn params(&self) -> &DeepModelParams<T> {
        &self.params
    }
    /// returns a mutable reference to the model parameters
    pub const fn params_mut(&mut self) -> &mut DeepModelParams<T> {
        &mut self.params
    }
    /// set the current configuration and return a mutable reference to the model
    pub fn set_config(&mut self, config: StandardModelConfig<T>) -> &mut Self {
        self.config = config;
        self
    }
    /// set the current features and return a mutable reference to the model
    pub fn set_features(&mut self, features: ModelFeatures) -> &mut Self {
        self.features = features;
        self
    }
    /// set the current parameters and return a mutable reference to the model
    pub fn set_params(&mut self, params: DeepModelParams<T>) -> &mut Self {
        self.params = params;
        self
    }
    /// consumes the current instance to create another with the given configuration
    pub fn with_config(self, config: StandardModelConfig<T>) -> Self {
        Self { config, ..self }
    }
    /// consumes the current instance to create another with the given features
    pub fn with_features(self, features: ModelFeatures) -> Self {
        Self { features, ..self }
    }
    /// consumes the current instance to create another with the given parameters
    pub fn with_params(self, params: DeepModelParams<T>) -> Self {
        Self { params, ..self }
    }
    /// initializes the model with Glorot normal distribution
    #[cfg(feature = "rand")]
    pub fn init(self) -> Self
    where
        T: Float + FromPrimitive,
        cnc::rand_distr::StandardNormal: cnc::rand_distr::Distribution<T>,
    {
        let params = DeepModelParams::glorot_normal(self.features());
        SimpleModel { params, ..self }
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

    fn forward(&self, input: &ArrayBase<S, D>) -> cnc::Result<Self::Output> {
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
        Ok(y)
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
    ) -> Result<Self::Output, ModelError> {
        if input.len() != self.features().input() {
            return Err(ModelError::InvalidInputShape);
        }
        if target.len() != self.features().output() {
            return Err(ModelError::InvalidOutputShape);
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
    ) -> Result<Self::Output, ModelError> {
        if input.nrows() == 0 || target.nrows() == 0 {
            return Err(ModelError::InvalidBatchSize);
        }
        if input.ncols() != self.features().input() {
            return Err(ModelError::InvalidInputShape);
        }
        if target.ncols() != self.features().output() || target.nrows() != input.nrows() {
            return Err(ModelError::InvalidOutputShape);
        }
        let mut loss = A::zero();

        for (i, (x, e)) in input.rows().into_iter().zip(target.rows()).enumerate() {
            loss += match Train::<ArrayView1<A>, ArrayView1<A>>::train(self, &x, &e) {
                Ok(l) => l,
                Err(err) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        "Training failed for batch {}/{}: {:?}",
                        i + 1,
                        input.nrows(),
                        err
                    );
                    #[cfg(not(feature = "tracing"))]
                    eprintln!(
                        "Training failed for batch {}/{}: {:?}",
                        i + 1,
                        input.nrows(),
                        err
                    );
                    return Err(err);
                }
            };
        }

        Ok(loss)
    }
}
