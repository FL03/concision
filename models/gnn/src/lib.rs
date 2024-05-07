/*
   Appellation: concision-gnn <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Graph Neural Networks (GNN)
//!
//! This library implements the framework for building graph-based neural networks.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_gnn"]

#[cfg(no_std)]
extern crate alloc;
extern crate concision_core as concision;
extern crate ndarray as nd;
#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;
extern crate ndarray_stats as stats;

pub mod prelude {}
