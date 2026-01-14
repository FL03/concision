# concision (cnc)

[![crates.io](https://img.shields.io/crates/v/concision?style=for-the-badge&logo=rust)](https://crates.io/crates/concision)
[![docs.rs](https://img.shields.io/docsrs/concision?style=for-the-badge&logo=docs.rs)](https://docs.rs/concision)
[![GitHub License](https://img.shields.io/github/license/FL03/concision?style=for-the-badge&logo=github)](https://github.com/FL03/concision/blob/main/LICENSE)

***

_**Warning: The library still in development and is not yet ready for production use.**_

**Note:** It is important to note that a primary consideration of the `concision` framework is ensuring compatibility in two key areas:

- `autodiff`: the upcoming feature enabling rust to natively support automatic differentiation.
- [`ndarray`](https://docs.rs/ndarray): The crate is designed around the `ndarray` crate, which provides a powerful N-dimensional array type for Rust

## Overview

### Goals

- Provide a flexible and extensible framework for building neural network models in Rust.
- Support both shallow and deep neural networks with a focus on modularity and reusability.
- Enable easy integration with other libraries and frameworks in the Rust ecosystem.

### Roadmap

- Implement additional optimization algorithms (e.g., Adam, RMSProp).
- Add support for convolutional and recurrent neural networks.
- Expand the set of built-in layers and activation functions.
- Improve documentation and provide more examples and tutorials.
- Implement support for automatic differentiation using the `autodiff` crate.

## Getting Started

### Adding to your project

To use `concision` in your project, run the following command:

```bash
cargo add concision --features full
```

or add the following to your `Cargo.toml`:

```toml
[dependencies.concision]
features = ["full"]
version = "0.3.x"
```

### Examples

#### **Example (1):** Simple Model

```rust
  use crate::activate::{ReLUActivation, SigmoidActivation};
  use crate::{
      DeepModelParams, Error, Forward, Model, ModelFeatures, Norm, Params, StandardModelConfig, Train,
  };
  #[cfg(feature = "rand")]
  use concision_init::{
      InitTensor,
      rand_distr::{Distribution, StandardNormal},
  };

  use ndarray::prelude::*;
  use ndarray::{Data, ScalarOperand};
  use num_traits::{Float, FromPrimitive, NumAssign, Zero};

  #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
  pub struct TestModel<T = f64> {
      pub config: StandardModelConfig<T>,
      pub features: ModelFeatures,
      pub params: DeepModelParams<T>,
  }

  impl<T> TestModel<T> {
      pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self
      where
          T: Clone + Zero,
      {
          let params = DeepModelParams::zeros(features);
          TestModel {
              config,
              features,
              params,
          }
      }
      /// returns an immutable reference to the model configuration
      pub const fn config(&self) -> &StandardModelConfig<T> {
          &self.config
      }
      /// returns a reference to the model layout
      pub const fn features(&self) -> ModelFeatures {
          self.features
      }
      /// returns a reference to the model params
      pub const fn params(&self) -> &DeepModelParams<T> {
          &self.params
      }
      /// returns a mutable reference to the model params
      pub const fn params_mut(&mut self) -> &mut DeepModelParams<T> {
          &mut self.params
      }
      #[cfg(feature = "rand")]
      /// consumes the current instance to initalize another with random parameters
      pub fn init(self) -> Self
      where
          StandardNormal: Distribution<T>,
          T: Float,
      {
          let TestModel {
              mut params,
              config,
              features,
          } = self;
          params.set_input(Params::<T>::lecun_normal((
              features.input(),
              features.hidden(),
          )));
          for layer in params.hidden_mut() {
              *layer = Params::<T>::lecun_normal((features.hidden(), features.hidden()));
          }
          params.set_output(Params::<T>::lecun_normal((
              features.hidden(),
              features.output(),
          )));
          TestModel {
              config,
              features,
              params,
          }
      }
  }

  impl<T> Model<T> for TestModel<T> {
      type Config = StandardModelConfig<T>;

      type Layout = ModelFeatures;

      fn config(&self) -> &StandardModelConfig<T> {
          &self.config
      }

      fn config_mut(&mut self) -> &mut StandardModelConfig<T> {
          &mut self.config
      }

      fn layout(&self) -> &ModelFeatures {
          &self.features
      }

      fn params(&self) -> &DeepModelParams<T> {
          &self.params
      }

      fn params_mut(&mut self) -> &mut DeepModelParams<T> {
          &mut self.params
      }
  }

  impl<A, S, D> Forward<ArrayBase<S, D, A>> for TestModel<A>
  where
      A: Float + FromPrimitive + ScalarOperand,
      D: Dimension,
      S: Data<Elem = A>,
      Params<A>: Forward<ArrayBase<S, D, A>, Output = Array<A, D>>
          + Forward<Array<A, D>, Output = Array<A, D>>,
  {
      type Output = Array<A, D>;

      fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
          // complete the first forward pass using the input layer
          let mut output = self.params().input().forward(input).relu();
          // complete the forward pass for each hidden layer
          for layer in self.params().hidden() {
              output = layer.forward(&output).relu();
          }

          self.params().output().forward(&output).sigmoid()
      }
  }

  impl<A, S, T> Train<ArrayBase<S, Ix1>, ArrayBase<T, Ix1>> for TestModel<A>
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
              return Err(Error::InvalidInputFeatures(
                  input.len(),
                  self.layout().input(),
              ));
          }
          if target.len() != self.layout().output() {
              return Err(Error::InvalidTargetFeatures(
                  target.len(),
                  self.layout().output(),
              ));
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

          let mut output = self.params().input().forward_then(&input, |y| y.relu());
          activations.push(output.to_owned());
          // collect the activations of the hidden
          for layer in self.params().hidden() {
              output = layer.forward(&output).relu();
              activations.push(output.to_owned());
          }

          output = self.params().output().forward(&output).sigmoid();
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
              .backward(activations.last().unwrap(), &delta, lr);

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
              self.params_mut().hidden_mut()[i].backward(&activations[i + 1], &delta, lr);
          }
          /*
              The delta for the input layer is computed using the weights of the first hidden layer
              and the derivative of the activation function of the first hidden layer.
          */
          delta = self.params().hidden()[0].weights().dot(&delta) * activations[1].relu_derivative();
          delta /= delta.l2_norm(); // Normalize the delta to prevent exploding gradients
          self.params_mut()
              .input_mut()
              .backward(&activations[1], &delta, lr);

          Ok(loss)
      }
  }

  impl<A, S, T> Train<ArrayBase<S, Ix2>, ArrayBase<T, Ix2>> for TestModel<A>
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
      ) -> Result<Self::Output, Self::Error> {
          if input.nrows() == 0 || target.nrows() == 0 {
              return Err(anyhow::anyhow!("Input and target batches must be non-empty").into());
          }
          if input.ncols() != self.layout().input() {
              return Err(Error::InvalidInputFeatures(
                  input.ncols(),
                  self.layout().input(),
              ));
          }
          if target.ncols() != self.layout().output() || target.nrows() != input.nrows() {
              return Err(Error::InvalidTargetFeatures(
                  target.ncols(),
                  self.layout().output(),
              ));
          }
          let batch_size = input.nrows();
          let mut loss = A::zero();

          for (i, (x, e)) in input.rows().into_iter().zip(target.rows()).enumerate() {
              loss += match Train::<ArrayView1<A>, ArrayView1<A>>::train(self, &x, &e) {
                  Ok(l) => l,
                  Err(err) => {
                      #[cfg(not(feature = "tracing"))]
                      eprintln!(
                          "Training failed for batch {}/{}: {:?}",
                          i + 1,
                          batch_size,
                          err
                      );
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

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. View the [quickstart guide](QUICKSTART.md) for more information on setting up your environment to develop the `concision` framework.

Please make sure to update tests as appropriate.

## License

- [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
- [MIT](https://choosealicense.com/licenses/mit/)
