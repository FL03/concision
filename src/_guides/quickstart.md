---
description: "A detailed guide to setting up and developing with the concision framework."
title: Quickstart
layout: default
nav_order: 2
---
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
