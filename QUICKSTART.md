# Quickstart

Welcome to the `concision` quickstart guide! This guide will help you get started with the `concision` crate, a complete machine learning framework for Rust. It provides a wide range of features for building and training machine learning models, including support for various data types, optimizers, and loss functions.

**Note:** It is important to note that a primary consideration of the `concision` framework is ensuring compatibility in two key areas:

- `autodiff`: the upcoming feature enabling rust to natively support automatic differentiation.
- [`ndarray`](https://docs.rs/ndarray): The crate is designed around the `ndarray` crate, which provides a powerful N-dimensional array type for Rust

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
