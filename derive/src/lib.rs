/*
   Appellation: concision-derive <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Derive macros for the concision framework.
//!
//! ## Overview
//!
//!
#![crate_type = "proc-macro"]

extern crate proc_macro;
extern crate quote;
extern crate syn;

pub(crate) mod ast;
#[allow(dead_code)]
pub(crate) mod attrs;
pub(crate) mod impls;
pub(crate) mod utils;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// The [`Configuration`] derive macro generates configuration-related code for a given struct, 
/// streamlining the process of creating compatible configuration spaces within the concision 
/// framework.
#[proc_macro_derive(Configuration, attributes(config))]
pub fn configuration(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let res = impls::impl_config(&input);

    // Return the generated code as a TokenStream
    res.into()
}

/// This macro generates a parameter struct and an enum of parameter keys.
#[proc_macro_derive(Keyed, attributes(keys))]
pub fn keyed(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let res = impls::impl_keys(&input);

    // Return the generated code as a TokenStream
    res.into()
}
