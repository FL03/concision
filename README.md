# concision (cnc)

[![crates.io](https://img.shields.io/crates/v/concision?style=for-the-badge&logo=rust)](https://crates.io/crates/concision)
[![docs.rs](https://img.shields.io/docsrs/concision?style=for-the-badge&logo=docs.rs)](https://docs.rs/concision)
[![GitHub License](https://img.shields.io/github/license/FL03/concision?style=for-the-badge&logo=github)](https://github.com/FL03/concision/blob/main/LICENSE)

***

_**Warning: The library still in development and is not yet ready for production use.**_

**Note:** It is important to note that a primary consideration of the `concision` framework is ensuring compatibility in two key areas:

- `autodiff`: the upcoming feature enabling rust to natively support automatic differentiation.
- [`ndarray`](https://docs.rs/ndarray): The crate is designed around the `ndarray` crate, which provides a powerful N-dimensional array type for Rust

## Usage

### Adding to your project

To use `concision` in your project, add the following to your `Cargo.toml`:

```toml
[dependencies.concision]
features = ["full"]
version = "0.1.x"
```

### Examples

#### **Example (1):** Simple Model

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

## Getting Started

### Prerequisites

To use `concision`, you need to have the following installed:

- [Rust](https://www.rust-lang.org/tools/install) (version 1.85 or later)

### Installation

You can install the `rustup` toolchain using the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installing `rustup`, you can install the latest stable version of Rust with:

```bash
rustup install stable
```

You can also install the latest nightly version of Rust with:

```bash
rustup install nightly
```

### Building from the source

Start by cloning the repository

```bash
git clone https://github.com/FL03/concision.git
```

Then, navigate to the `concision` directory:

```bash
cd concision
```

#### _Using the `cargo` tool_

To build the crate, you can use the `cargo` tool. The following command will build the crate with all features enabled:

```bash
cargo build -r --locked --workspace --features full
```

To run the tests, you can use the following command:

```bash
cargo test -r --locked --workspace --features full
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

- [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
- [MIT](https://choosealicense.com/licenses/mit/)
