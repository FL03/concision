/*
    appellation: node <module>
    authors: @FL03
*/
use crate::ast::ConfigAst;
use proc_macro2::TokenStream;
use quote::quote;

/// this method defines the logic enabling the `model_config!` procedural macro.
/// it takes a `ConfigBlock` and generates the necessary code to define the model configuration
pub fn impl_model_config(_config: ConfigAst) -> TokenStream {
    // generate the output code
    quote! {}
}
