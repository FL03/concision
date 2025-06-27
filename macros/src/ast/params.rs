/*
    appellation: params <module>
    authors: @FL03
*/
use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::token::Brace;
use syn::{Expr, Ident, braced};

#[allow(dead_code)]
pub struct ParamsBlock {
    pub key: kw::params,
    pub brace_token: Brace,
    pub contents: Vec<ParamStmt>,
}

#[allow(dead_code)]
pub struct ParamStmt {
    pub key: Ident,
    pub value: syn::Expr,
}

impl Parse for ParamStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        let key: Ident = input.parse()?;
        // parse the equals sign
        input.parse::<syn::Token![=]>()?;
        // parse the expression
        let value: Expr = input.parse()?;
        // create the ParamStmt instance
        Ok(ParamStmt { key, value })
    }
}

impl Parse for ParamsBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `params` keyword
        let key: kw::params = input.parse()?;
        // create a buffer for the braced content
        let content;
        let brace_token = braced!(content in input);
        // create a store for the parsed statements
        let mut contents = Vec::new();
        // parse the braced content until it is empty
        while !content.is_empty() {
            let stmt: ParamStmt = content.parse()?;
            contents.push(stmt);
        }
        let res = ParamsBlock {
            key,
            brace_token,
            contents,
        };
        Ok(res)
    }
}
