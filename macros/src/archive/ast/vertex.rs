/*
    appellation: vertex <ast>
    authors: @FL03
*/
use quote::ToTokens;
use syn::parse::{Parse, ParseStream};
use syn::{token, Ident};

/// The [`VertexAst`] is an abstract syntax tree (AST) representation for the definition of a 
/// vertex in a hypergraph. It can essentially be thought of as the beginning of a standard 
/// `let` statement.
/// 
/// ```ignore
/// let v0;
/// ```
/// 
/// or 
/// 
/// ```ignore
/// let v1 = 50;
/// ```
#[allow(dead_code)]
pub struct VertexAst {
    pub let_token: token::Let,
    pub key: Ident,
    pub colon_token: Option<token::Colon>,
}

impl Parse for VertexAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let let_token = input.parse()?;
        let key = input.parse()?;
        // handle the case where a weight is defined
        let colon_token = if input.peek(token::Colon) {
            Some(input.parse()?)
        } else {
            None
        };

        Ok(Self { let_token, key, colon_token })
    }
}

impl ToTokens for VertexAst {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.let_token.to_tokens(tokens);
        self.key.to_tokens(tokens);
        // handle the case where the colon_token is present
        if let Some(colon) = &self.colon_token {
            colon.to_tokens(tokens);
        }
    }
}