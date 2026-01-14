/*
    Appellation: impls <module>
    Created At: 2025.12.07:11:55:02
    Contrib: @FL03
*/

mod impl_config;
mod impl_keys;

use proc_macro2::TokenStream;
use syn::{Data, DeriveInput};

pub fn impl_config(DeriveInput { ident, data, .. }: &DeriveInput) -> TokenStream {
    match &data {
        Data::Struct(s) => impl_config::derive_config_from_struct(s, ident),
        _ => panic!("Only structs are supported"),
    }
}

pub fn impl_keys(DeriveInput { data, ident, .. }: &DeriveInput) -> TokenStream {
    match &data {
        Data::Struct(s) => impl_keys::generate_keys_for_struct(s, ident),

        _ => panic!("Only structs are supported"),
    }
}
