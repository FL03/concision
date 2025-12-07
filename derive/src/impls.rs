/*
    Appellation: impls <module>
    Created At: 2025.12.07:11:55:02
    Contrib: @FL03
*/

mod impl_config;
mod impl_keys;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DataStruct, DeriveInput};

pub fn impl_config(input: &DeriveInput) -> TokenStream {
    // Get the name of the struct
    let struct_name = &input.ident;
    let store_name = format_ident!("{}Config", struct_name);

    // Generate the parameter struct definition

    // Generate the parameter keys enum
    let param_keys_enum = match &input.data {
        Data::Struct(s) => {
            let DataStruct { fields, .. } = s;

            impl_config::generate_config(fields, &store_name)
        }
        _ => panic!("Only structs are supported"),
    };

    // Combine the generated code
    quote! {
        #param_keys_enum
    }
}

pub fn impl_keys(input: &DeriveInput) -> TokenStream {
    // Get the name of the struct
    let struct_name = &input.ident;
    let store_name = format_ident!("{}Key", struct_name);

    // Generate the parameter struct definition

    // Generate the parameter keys enum
    let param_keys_enum = match &input.data {
        Data::Struct(s) => {
            let DataStruct { fields, .. } = s;

            impl_keys::generate_keys(fields, &store_name)
        }
        _ => panic!("Only structs are supported"),
    };

    // Combine the generated code
    quote! {
        #param_keys_enum
    }
}
