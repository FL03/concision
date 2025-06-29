/*
    appellation: graph <ast>
    authors: @FL03
*/
use crate::kw;
use crate::ast::WeightAst;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, Ident, Token, braced, bracketed, token};


/// The [`EdgeStmt`] struct represents the abstract syntax tree (AST) for defining edges in a
/// hypergraph.
///
/// ```ignore
/// edges: {
///     let e0: [v0, v1];
///     let e1: [v0, v1, v2] = 50;
/// };
/// ```
#[allow(dead_code)]
pub struct EdgeBlock {
    pub attrs: Vec<Attribute>,
    pub block_id: kw::model,
    pub colon_token: token::Colon,
    pub brace_token: token::Brace,
    pub contents: Punctuated<EdgeStmt, token::Semi>,
    pub semi_token: token::Semi,
}
/// The [`EdgeStmt`] struct represents the abstract syntax tree (AST) for defining edges in a
/// hypergraph.
///
/// ```ignore
/// let e0: [v0, v1];
/// ```
///
/// or
///
/// ```ignore
/// let e1: [v0, v1, v2] = 50;
/// ```
#[allow(dead_code)]
pub struct EdgeStmt {
    pub attrs: Vec<Attribute>,
    pub let_token: token::Let,
    pub key: Ident,
    pub colon_token: token::Colon,
    pub bracket_token: token::Bracket,
    pub domain: Punctuated<Ident, token::Comma>,
    pub weight: Option<WeightAst>,
    pub semi_token: token::Semi,
}

impl Parse for EdgeStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the outer attributes
        let attrs = input.call(Attribute::parse_outer)?;
        // parse the "let" keyword
        let let_token = input.parse()?;
        // parse the identifier for the edge index
        let key = input.parse()?;
        // parse the colon token
        let colon_token = input.parse()?;
        // initialize a buffer for the contents of the brackets
        let content;
        // parse the bracket token
        let bracket_token = bracketed!(content in input);
        // parse the bracketed content, which should be a list of node identifiers
        let domain = content.parse_terminated(Ident::parse, token::Comma)?;
        // if defined, parse the weight of the edge
        let weight = if input.peek(token::Eq) {
            Some(input.parse::<WeightAst>()?)
        } else {
            None
        };
        // parse the end-of-statement semicolon
        let semi_token = input.parse::<Token![;]>()?;
        // finish by returning the parse ast
        Ok(EdgeStmt {
            attrs,
            let_token,
            key,
            colon_token,
            bracket_token,
            domain,
            weight,
            semi_token
        })
    }
}

impl Parse for EdgeBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the outer attributes
        let attrs = input.call(Attribute::parse_outer)?;
        // parse the "edges" keyword
        let edges_token = input.parse::<kw::edges>()?;
        // colon token is required after the edges keyword
        let colon_token = input.parse::<token::Colon>()?;
        // parse the opening brace
        let buf;
        // extract the content within the braces
        let brace_token = braced!(buf in input);
        // parse the edge statements
        let contents = buf.parse_terminated(EdgeStmt::parse, token::Semi)?;
        // parse the end-of-block semicolon
        let semi_token = input.parse::<token::Semi>()?;
        // return the parsed edge block
        Ok(EdgeBlock {
            attrs,
            edges_token,
            colon_token,
            brace_token,
            contents,
            semi_token,
        })
    }
}
