/*
    Appellation: impl_config <module>
    Created At: 2025.12.07:11:54:21
    Contrib: @FL03
*/
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, FieldsNamed, FieldsUnnamed, Ident};

pub fn derive_config_from_struct(
    DataStruct { fields, .. }: &DataStruct,
    name: &Ident,
) -> TokenStream {
    match fields {
        Fields::Named(inner) => handle_named(inner, name),
        Fields::Unnamed(inner) => handle_unnamed(inner, name),
        _ => panic!("Unit fields aren't currently supported."),
    }
}

fn handle_named(_fields: &FieldsNamed, name: &Ident) -> TokenStream {
    // let FieldsNamed { named, .. } = fields;

    quote! {
        impl #name {

        }
    }
}

fn handle_unnamed(_fields: &FieldsUnnamed, name: &Ident) -> TokenStream {
    quote! {
        impl #name {

        }
    }
}
