/*
   Appellation: concision-derive <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Derive macros for the concision framework.
//!
//! ## Overview
//!
//!
#![crate_name = "concision_derive"]
#![crate_type = "proc-macro"]

extern crate proc_macro;
extern crate quote;
extern crate syn;

pub(crate) mod ast;
pub(crate) mod attrs;
pub(crate) mod params;
pub(crate) mod utils;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// This macro generates a parameter struct and an enum of parameter keys.
#[proc_macro_derive(Keyed, attributes(param))]
pub fn keyed(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let res = params::impl_params(&input);

    // Return the generated code as a TokenStream
    res.into()
}
