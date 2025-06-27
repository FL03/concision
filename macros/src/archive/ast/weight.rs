/*
    appellation: node <ast>
    authors: @FL03
*/
use syn::parse::{Parse, ParseStream};
use syn::{Expr, token};

/// the [`WeightAst`] struct is used to generically represent the weight of an edge or node 
/// within a hypergraph. More specifically, it is used to define the tail end of a node or edge
/// statement composed of:
/// 
/// - `eq_token`: An equals sign (`=`) that indicates the start of the weight definition.
/// - `expr`: An expression that represents the weight value, which can be any valid Rust 
///   expression.
#[allow(dead_code)]
pub struct WeightAst {
    pub eq_token: token::Eq,
    pub expr: Box<Expr>,
}

impl Parse for WeightAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let eq_token = input.parse()?;
        let expr = input.parse()?;

        Ok(Self {
            eq_token,
            expr: Box::new(expr),
        })
    }
}

impl quote::ToTokens for WeightAst {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.eq_token.to_tokens(tokens);
        self.expr.to_tokens(tokens);
    }
}
