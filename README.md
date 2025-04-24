# concision (cnc)

[![license](https://img.shields.io/crates/l/concision.svg)](https://choosealicense.com/licenses/apache-2.0/)
[![crates.io](https://img.shields.io/crates/v/concision.svg)](https://crates.io/crates/concision)
[![docs.rs](https://docs.rs/concision/badge.svg)](https://docs.rs/concision)

[![clippy](https://github.com/FL03/concision/actions/workflows/clippy.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/clippy.yml)
[![rust](https://github.com/FL03/concision/actions/workflows/rust.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/rust.yml)

***

_**Warning: The library is currently in the early stages of development and is not yet ready for production use.**_

Concision is designed to be a complete toolkit for building machine learning models in Rust.

Concision is a machine learning library for building powerful models in Rust prioritizing ease-of-use, efficiency, and flexability. The library is built to make use of the both the upcoming `autodiff` experimental feature and increased support for generics in the 2024 edition of Rust.

## Getting Started

### Building from the source

Start by cloning the repository

```bash
git clone https://github.com/FL03/concision.git
cd concision
```

```bash
cargo build --features full -r --workspace
```

### Testing the crate

```bash
cargo test --workspace -F full
```

## Usage

### Example: Creating a simple Model

```rust
    extern crate concision as cnc;

    use cnc::activate::{ReLU, Sigmoid};
    use cnc::nn::{Model, ModelFeatures, ModelParams, StandardModelConfig};
    use ndarray::{Array1, ScalarOperand};
    use num::Float;

    pub struct SimpleModel<T = f64> {
        pub config: StandardModelConfig<T>,
        pub features: ModelFeatures,
        pub params: ModelParams<T>,
    }

    impl<T> SimpleModel<T> {
        pub fn new(config: StandardModelConfig<T>, features: ModelFeatures) -> Self 
        where 
            T: Clone + num::Zero
        {
            let params = ModelParams::zeros(features);
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

        fn params(&self) -> &ModelParams<T> {
            &self.params
        }

        fn params_mut(&mut self) -> &mut ModelParams<T> {
            &mut self.params
        }
    }
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

* [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
* [MIT](https://choosealicense.com/licenses/mit/)
