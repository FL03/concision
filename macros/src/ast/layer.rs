/*
    appellation: layer <module>
    authors: @FL03
*/
use crate::kw;
use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::token::{Brace, Colon, Eq, Semi};
use syn::{Expr, Ident, Type, braced};

/// The [`LayerBlock`] struct represents a block of layer definitions in the model DSL.
///
/// ```no_run
/// model! {
///     layer: {
///         ... // various layer statements
///     }
/// }
/// ```
#[allow(dead_code)]
pub struct LayerBlock {
    pub key: kw::layer,
    pub brace_token: Brace,
    pub contents: Vec<LayerStmt>,
    pub eos: Option<Semi>,
}
/// The [`LayerStmt`] defines the structure of a single layer statement within a [`LayerBlock`]
///
/// ```no_run
/// let Layer1: ty = LayerExpr;
/// ```
#[allow(dead_code)]
pub struct LayerStmt {
    pub key: Ident,
    pub colon_token: Colon,
    pub type_token: Type,
    pub eq_token: Eq,
    /// the layer expression, which can be an identifier, an expression, or a raw token stream
    pub layer: LayerExpr,
    /// the end-of-statement token
    pub eos: Semi,
}

#[allow(dead_code)]
pub enum LayerExpr {
    Expr(Expr),
    /// Represents a layer expression that can be used in the model DSL.
    Named(Ident),
    /// raw, undefined layer expression
    Raw(TokenStream),
}

impl Parse for LayerExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        if input.peek(Ident) {
            let key = input.parse::<Ident>()?;
            return Ok(LayerExpr::Named(key));
        }
        // fallback onto the raw variant for any undefined behavior
        Ok(Self::Raw(input.parse()?))
    }
}

impl Parse for LayerStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        let key = input.parse::<Ident>()?;
        // parse the domain expression
        let colon_token = input.parse::<Colon>()?;
        let type_token = input.parse::<Type>()?;
        let eq_token = input.parse::<Eq>()?;
        let layer = input.parse::<LayerExpr>()?;
        let eos = input.parse::<Semi>()?;

        // create the LayerStmt instance
        Ok(LayerStmt {
            key,
            colon_token,
            type_token,
            eq_token,
            layer,
            eos,
        })
    }
}

impl Parse for LayerBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `layer` keyword
        let key = input.parse::<kw::layer>()?;
        // create a store for the layer block
        let mut contents = Vec::new();
        // create a buffer for the braced content
        let content; // the layer {} outer brace
        let brace_token = braced!(content in input); // parse the brace
        while !content.is_empty() {
            let stmt: LayerStmt = content.parse()?;
            contents.push(stmt);
        }
        let eos = if content.peek(Semi) {
            Some(content.parse::<Semi>()?)
        } else {
            None
        };
        Ok(LayerBlock {
            key,
            brace_token,
            contents,
            eos,
        })
    }
}
