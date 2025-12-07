/*
    Appellation: impl_config <module>
    Created At: 2025.12.07:11:54:21
    Contrib: @FL03
*/
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Fields, FieldsNamed, Ident};

pub fn generate_config(fields: &Fields, name: &Ident) -> TokenStream {
    match fields {
        Fields::Named(inner) => handle_named(inner, name),
        _ => panic!("Only named fields are supported"),
    }
}

pub fn handle_named(fields: &FieldsNamed, name: &Ident) -> TokenStream {
    let FieldsNamed { named, .. } = fields;

    quote! {

        impl #name {
        }
    }
}
