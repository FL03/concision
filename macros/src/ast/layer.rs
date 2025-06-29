/*
    appellation: layer <module>
    authors: @FL03
*/
use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::token::{Brace, Colon};
use syn::{Expr, Ident, braced};

#[allow(dead_code)]
pub struct LayerBlock {
    pub key: kw::layer,
    pub contents: Vec<LayerStmt>,
    pub brace_token: Brace,
}

#[allow(dead_code)]
pub struct LayerStmt {
    pub key: Ident,
    pub colon_token: Colon,
    pub domain: Expr,
}

impl Parse for LayerStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        let key = input.parse::<Ident>()?;
        // parse the domain expression
        let colon_token = input.parse::<Colon>()?;
        let domain = input.parse::<Expr>()?;
        // create the LayerStmt instance
        Ok(LayerStmt {
            key,
            colon_token,
            domain,
        })
    }
}

impl Parse for LayerBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `layer` keyword
        let key: kw::layer = input.parse()?;
        // create a store for the layer block
        let mut contents = Vec::new();
        // create a buffer for the braced content
        let content; // the layer {} outer brace
        let brace_token = braced!(content in input); // parse the brace
        while !content.is_empty() {
            let stmt: LayerStmt = content.parse()?;
            contents.push(stmt);
        }
        Ok(LayerBlock {
            key,
            contents,
            brace_token,
        })
    }
}
