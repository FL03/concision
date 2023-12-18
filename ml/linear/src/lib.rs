/*
   Appellation: concision-linear <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-linear
//!
//! This library implements the framework for building linear models.
//!

pub mod conv;
pub mod dense;
pub mod model;
pub mod params;

pub(crate) use concision_core as core;
pub(crate) use concision_neural as neural;

pub mod prelude {}
