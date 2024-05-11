/*
   Appellation: concision-kan <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Kolmogorov-Arnold Networks (KAN)
//!
//! Kolmogorov-Arnold Networks (KAN) are a novel class of neural networks based on the Kolmogorov-Arnold Representation Theorem.
//! These models have already demonstrated that they are viable alternatives to traditional multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs).
//!
//! ### Resources
//! 
//! - [Kolmogorov-Arnold Representation](https://arxiv.org/abs/2404.19756)
//! 
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate concision_core as concision;
extern crate ndarray as nd;

pub mod prelude {}
