/*
    appellation: model <ast>
    authors: @FL03
*/
use super::{LayerBlock, ParamsBlock};

use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, braced, token};

#[allow(dead_code)]
/// The [`ModelAst`] struct represents the abstract syntax tree (AST) for a model macro.
/// Overall, it is supposed to feel similar to
///
/// ```ignore
/// model! {
///     model {
///         config: {
///             decay: f64;
///             learning_rate: f64 = 0.001;
///             ... // declare other hyperparameters here; optionally, set their default values
///         }
///         params: {
///             ...
///         };
///         layers: {
///             ...
///         };
///     }
/// }
/// ```
pub struct ModelAst {
    pub model: Ident,
    pub brace_token: token::Brace,
    pub params_block: Option<ParamsBlock>,
    pub layer_block: Option<LayerBlock>,
    pub semi_token: Option<token::Semi>,
}

impl Parse for ModelAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let model: Ident = input.parse()?;
        let content; // the model {} outer brace
        let brace_token = braced!(content in input);

        let mut params_block: Option<ParamsBlock> = None;
        let mut layer_block: Option<LayerBlock> = None;

        while !content.is_empty() {
            // if present, parse the layer block
            if content.peek(kw::layer) {
                // parse the layer block
                layer_block = content.parse().ok();
            } else if content.peek(kw::params) {
                // parse the nodes block
                params_block = content.parse().ok();
            } else {
                break;
            }
        }

        // parse the final semi token, if provided
        let semi_token = if input.peek(token::Semi) {
            input.parse().ok()
        } else {
            None
        };
        // ensure that either just the nodes or both nodes and edges are present since we can
        // only interact with code that is "in-scope" of the macro
        if layer_block.is_some() && params_block.is_none() {
            return Err(syn::Error::new_spanned(
                model,
                "To define any edges with the macro, you must also define nodes.",
            ));
        }
        // return the parsed model AST
        Ok(ModelAst {
            model,
            brace_token,
            layer_block,
            params_block,
            semi_token,
        })
    }
}
