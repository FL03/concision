/*
    appellation: node <module>
    authors: @FL03
*/
use crate::ast::{LayerBlock, LayerStmt, ModelAst, ParamStmt, ParamsBlock};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub fn impl_model(
    ModelAst {
        model,
        layer_block: layers,
        params_block: params,
        ..
    }: ModelAst,
) -> TokenStream {
    // if there are no nodes or edges, return an empty TokenStream
    if params.is_none() && layers.is_none() {
        return TokenStream::new();
    }
    // if edges are defined, but no nodes, return an error
    if layers.is_some() && params.is_none() {
        return syn::Error::new_spanned(model, "edges cannot be defined without nodes")
            .to_compile_error();
    }
    // initialize a vector to hold the statements
    let mut stmts = Vec::new();
    // handle the node block
    if let Some(ParamsBlock { contents, .. }) = &params {
        contents.iter().for_each(|node| {
            let stmt = handle_params(&model, node);
            stmts.push(stmt);
        });
    }
    // handle the edge block
    if let Some(LayerBlock { contents, .. }) = &layers {
        contents.iter().for_each(|edge| {
            let stmt = handle_layer(&model, edge);
            stmts.push(stmt);
        });
    }
    // generate the output code
    quote! {
        #(#stmts)*
    }
}
/// a method for handling individual edge statements
pub fn handle_layer(_model: &Ident, _layer: &LayerStmt) -> TokenStream {
    // let LayerStmt { key, domain, .. } = layer;

    quote! {}
}

pub fn handle_params(_model: &Ident, _params: &ParamStmt) -> TokenStream {
    quote! {}
}
