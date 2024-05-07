# Concision

[![clippy](https://github.com/FL03/concision/actions/workflows/clippy.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/clippy.yml)
[![publish](https://github.com/FL03/concision/actions/workflows/publish.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/publish.yml)
[![rust](https://github.com/FL03/concision/actions/workflows/rust.yml/badge.svg)](https://github.com/FL03/concision/actions/workflows/rust.yml)

[![crates.io](https://img.shields.io/crates/v/concision.svg)](https://crates.io/crates/concision)
[![docs.rs](https://docs.rs/concision/badge.svg)](https://docs.rs/concision)

***

Concision is designed to be a complete toolkit for building machine learning models in Rust. 

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

## Usage

```rust
    extern crate concision as cnc;

    use cnc::func::Sigmoid;
    use cnc::linear::{Config, Features, Linear};
    use cnc::{linarr, Predict, Result};
    use ndarray::Ix2;

    fn main() -> Result<()> {
        tracing_subscriber::fmt::init();
        tracing::info!("Starting linear model example");

        let (samples, dmodel, features) = (20, 5, 3);
        let features = Features::new(3, 5);
        let config = Config::new("example", features).biased();
        let data = linarr::<f64, Ix2>((samples, dmodel)).unwrap();

        let model: Linear<f64> = Linear::std(config).uniform();
        // `.activate(*data, *activation)` runs the forward pass and applies the activation function to the result
        let y = model.activate(&data, Sigmoid::sigmoid).unwrap();
        assert_eq!(y.dim(), (samples, features));
        println!("Predictions: {:?}", y);

        Ok(())
    }
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

* [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
* [MIT](https://choosealicense.com/licenses/mit/)
