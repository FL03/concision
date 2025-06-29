---
description: Creating models with the concision framework
title: Parameters
layout: default
nav_order: 1
---

## Overview

The `ParamsBase` struct is a key component of the `cnc` framework, designed around the [`ndarray`](https://docs.rs/ndarray) crate. It serves as a base for creating parameter sets that can be used in various machine learning models and algorithms. This struct is useful in that it provides a flexible and efficient way to manage both weights and biases within a layer of a neural network or any other model that requires parameters. Additionally, it automatically handles the dimensionality of the parameters, ensuring that the `bias` tensor is always one dimension smaller than the `weights` tensor. Consequently, the smallest dimension the `ParamsBase` can handle is one dimension; in this case, the `bias` tensor will be a scalar, also known as a `rank(0)` tensor.

## Features

- `blas`: Enables the use of BLAS (Basic Linear Algebra Subprograms) for optimized linear
  algebra operations, which can significantly speed up computations involving matrices and
  vectors. That being said, it requires careful setup and linking to the appropriate BLAS
  library, which can vary depending on the system and the specific BLAS implementation used.
- `init`: Enables various initialization routines for the parameters
- `rand`: Enables random initialization of parameters using the `rand` crate and extends the
  available initializers
- `rayon`: Enables parallel processing capabilities using the `rayon` crate
- `serde`: Enables serialization and deserialization of parameters using the `serde` crate

## Examples

For more detailed examples, please refer to the [examples directory](https://github.com/FL03/concision/blob/main/concision/examples).

### _Example #1: Basic Usage_

```rust
extern crate concision as cnc;

use cnc::params::Params;

let (m, n) = (8, 9);

let inputs = Array1::linspace(0.0, 1.0, m);
// initialize a 2-dimensional parameter set with 8 samples and 9 features
let mut params = dbg!(Params::<f64>::default((m, n)));
// validate the shape of the parameters
assert_eq!(params.weights().shape(), &[m, n]);
assert_eq!(params.bias().shape(), &[n]);
// initialize the parameters with random values
params.assign_weights(&Array2::glorot_normal((m, n)));
params.assign_bias(&Array1::glorot_normal((n,)));
// validate the shape of the parameters
assert_eq!(params.weights().shape(), &[m, n]);
assert_eq!(params.bias().shape(), &[n]);

let y = dbg!(params.forward(&inputs)?);
assert_eq!(y.shape(), &[n]);
```
