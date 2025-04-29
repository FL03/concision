/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub mod keys;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DataStruct, DeriveInput};

pub fn impl_params(input: &DeriveInput) -> TokenStream {
    // Get the name of the struct
    let struct_name = &input.ident;
    let store_name = format_ident!("{}Key", struct_name);

    // Generate the parameter struct definition

    // Generate the parameter keys enum
    let param_keys_enum = match &input.data {
        Data::Struct(s) => {
            let DataStruct { fields, .. } = s;

            keys::generate_keys(fields, &store_name)
        }
        _ => panic!("Only structs are supported"),
    };

    // Combine the generated code
    quote! {
        #param_keys_enum
    }
}
