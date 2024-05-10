/*
   Appellation: concision-kan <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Kolmogorov-Arnold Networks (KAN)
//!
//!

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(no_std)]
extern crate alloc;
extern crate concision_core as concision;
extern crate ndarray as nd;

pub mod prelude {}
