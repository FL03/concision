# Concision

[![Clippy](https://github.com/scattered-systems/concision/actions/workflows/clippy.yml/badge.svg)](https://github.com/scattered-systems/concision/actions/workflows/clippy.yml)
[![Rust](https://github.com/scattered-systems/concision/actions/workflows/rust.yml/badge.svg)](https://github.com/scattered-systems/concision/actions/workflows/rust.yml)

***

Inspired by the myriad of data science libraries created for Python, concision is a complete data-science toolkit
written in Rust and designed to support the creation of enterprise-grade, data driven applications.

## Getting Started

### Building from the source

Start by cloning the repository

```bash
git clone https://github.com/scattered-systems/concision
```

```bash
cargo build --release --workspace
cargo test --all --all-features --release
```

## Usage

```rust
    use concision as cnc;

    fn main() {
        let a = "";

        println!("{:?}", a);
    }
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

* [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
* [MIT](https://choosealicense.com/licenses/mit/)
