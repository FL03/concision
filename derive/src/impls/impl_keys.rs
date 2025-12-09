/*
    Appellation: keys <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::utils::capitalize_first;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Fields, FieldsNamed, Ident, Variant};

pub fn generate_keys(fields: &Fields, name: &Ident) -> TokenStream {
    match fields {
        Fields::Named(inner) => handle_named(inner, name),
        _ => panic!("Only named fields are supported"),
    }
}

pub fn handle_named(fields: &FieldsNamed, name: &Ident) -> TokenStream {
    let FieldsNamed { named, .. } = fields;
    let methods = named.iter().cloned().map(|f| {
        let ident = f.ident.unwrap();
        let method = format_ident!("{}", &ident.to_string().to_lowercase());
        let variant = format_ident!("{}", capitalize_first(&ident.to_string()));
        quote! {
            pub fn #method() -> Self {
                Self::#variant
            }
        }
    });
    let variants = named.iter().cloned().map(|f| {
        let ident = f.ident.unwrap();
        let ident = format_ident!("{}", capitalize_first(&ident.to_string()));
        Variant {
            attrs: vec![],
            discriminant: None,
            fields: Fields::Unit,
            ident,
        }
    });

    quote! {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum #name {
            #(#variants),*
        }

        impl #name {
            #(#methods)*
        }
    }
}
