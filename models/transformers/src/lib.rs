/*
   Appellation: concision-transformers <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Transformers
//!
//! ### Resources
//!
//! - [Attention is All You Need](https://arxiv.org/abs/1706.03762)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;
extern crate ndarray as nd;

pub use self::transformer::Transformer;

pub(crate) mod transformer;

pub mod attention;

pub mod prelude {}
