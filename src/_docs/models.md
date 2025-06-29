---
description: Creating models with the concision framework
title: Models
layout: default
nav_order: 2
---

## Overview

The concision (`cnc`) framework provides a flexible and efficient way to create machine learning models. It is designed to work seamlessly with the [`ndarray`](https://docs.rs/ndarray) crate, allowing for easy manipulation of multi-dimensional arrays. The framework supports various model configurations, including deep neural networks, and provides a set of features for parameter management, activation functions, and forward propagation.

## Features

## Examples

For more detailed examples, please refer to the [examples directory](https://github.com/FL03/concision/blob/main/concision/examples).

### _Example #1: Basic Usage_

This example demonstrates how to create a simple model using the `cnc` framework. The model consists of an input layer, several hidden layers, and an output layer, with ReLU and Sigmoid activations applied at various stages.

```rust
    extern crate concision as cnc;

    use cnc::activate::{ReLU, Sigmoid};
    use cnc::nn::{Model, ModelFeatures, DeepModelParams, StandardModelConfig};
    use ndarray::{Array1, ScalarOperand};
    use num::Float;

    pub struct SimpleModel<T = f64> {
        pub config: StandardModelConfig<T>,
        pub features: ModelFeatures,
        pub params: DeepModelParams<T>,
    }

    impl<T> SimpleModel<T> {
        pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self 
        where 
            T: Clone + num::Zero
        {
            let params = DeepModelParams::zeros(features);
            SimpleModel {
                config,
                features,
                params,
            }
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

        fn params(&self) -> &DeepModelParams<T> {
            &self.params
        }

        fn params_mut(&mut self) -> &mut DeepModelParams<T> {
            &mut self.params
        }
    }
```
