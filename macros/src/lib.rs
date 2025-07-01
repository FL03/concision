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
pub(crate) mod kw;

use self::ast::ModelAst;
use proc_macro::TokenStream;

#[proc_macro]
/// [`model_config!`] is a procedural macro used to define the configuration for a model in the
/// `concision` framework. It allows users to specify various parameters and settings for the model
/// in a concise and structured manner, declaring a name for their instanc
pub fn model_config(input: TokenStream) -> TokenStream {
    let data = syn::parse_macro_input!(input as ast::ConfigAst);
    // use the handler to process the input data
    let res = impls::impl_model_config(data);
    // convert the tokens into a TokenStream
    res.into()
}

#[proc_macro]
/// the [`model!`]procedural macro is used to streamline the creation of custom models using the
/// `concision` framework
pub fn model(input: TokenStream) -> TokenStream {
    let data = syn::parse_macro_input!(input as ModelAst);
    // use the handler to process the input data
    let res = impls::impl_model(data);
    // convert the tokens into a TokenStream
    res.into()
}
