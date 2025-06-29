---
description: Usage guide for the `rshyper` library, including features and examples.
title: Usage Guide
layout: default
order: 1
---

## Getting Started

To use `concision` in your project, add the following to your `Cargo.toml`:

```toml
[dependencies.concision]
features = ["full"]
version = "0.2.x"
```

### Examples

#### **Example (1):** Simple Model

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
