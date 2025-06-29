/*
    appellation: concision-macros <library>
    authors: @FL03
*/
//! # concision-macros
//!
//! this crate defines various procedural macros for the `concision` crate working to
//! streamline the process of creating and developing custom neural networks.
//!

pub(crate) mod ast;
pub(crate) mod attr;
pub(crate) mod impls;

use self::ast::ModelAst;
use proc_macro::TokenStream;

#[doc(hidden)]
#[proc_macro]
/// the [`model!`]procedural macro is used to streamline the creation of custom models using the
/// `concision` framework
pub fn model(input: TokenStream) -> TokenStream {
    let data = syn::parse_macro_input!(input as ModelAst);
    // use the handler to process the input data
    let res = self::impls::impl_model(data);
    // convert the tokens into a TokenStream
    res.into()
}

mod kw {
    use syn::custom_keyword;
    custom_keyword! { model }
    custom_keyword! { layer }
    custom_keyword! { layers }
    custom_keyword! { param }
    custom_keyword! { params }
}
