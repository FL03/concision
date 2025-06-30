/*
    appellation: layer <module>
    authors: @FL03
*/
use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::token::{Brace, Colon, Eq, Semi};
use syn::{Expr, Ident, Type, braced};

/// The [`ConfigBlock`] struct represents a block of layer definitions in the model DSL.
///
/// ```no_run
/// model! {
///     config: {
///         ... // various layer statements
///     }
/// }
/// ```
#[allow(dead_code)]
pub struct ConfigBlock {
    pub key: kw::layer,
    pub brace_token: Brace,
    pub contents: Vec<ConfigStmt>,
    pub eos: Option<Semi>,
}
/// The [`ConfigStmt`] defines the structure of a single layer statement within a [`ConfigBlock`]
///
/// ```no_run
/// $key: $ty = $default;
/// ```
#[allow(dead_code)]
pub struct ConfigStmt {
    pub key: Ident,
    pub colon_token: Colon,
    pub typed: Type,
    /// the layer expression, which can be an identifier, an expression, or a raw token stream
    pub value: Option<ConfigExpr>,
    /// the end-of-statement token
    pub eos: Semi,
}
#[allow(dead_code)]
pub struct ConfigExpr {
    pub eq_token: Eq,
    /// the layer expression, which can be an identifier, an expression, or a raw token stream
    pub value: Expr,
}

impl Parse for ConfigExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the equals token
        let eq_token = input.parse::<Eq>()?;
        // parse the expression
        let value = input.parse::<Expr>()?;
        // create the ConfigExpr instance
        Ok(ConfigExpr { eq_token, value })
    }
}

impl Parse for ConfigStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        let key = input.parse::<Ident>()?;
        // parse the domain expression
        let colon_token = input.parse::<Colon>()?;
        // parse the type of the hyperparam
        let typed = input.parse::<Type>()?;
        //
        let value = input.parse::<ConfigExpr>().ok();
        let eos = input.parse::<Semi>()?;

        // create the ConfigStmt instance
        Ok(ConfigStmt {
            key,
            colon_token,
            typed,
            value,
            eos,
        })
    }
}

impl Parse for ConfigBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `layer` keyword
        let key = input.parse::<kw::layer>()?;
        // create a store for the layer block
        let mut contents = Vec::new();
        // create a buffer for the braced content
        let content; // the layer {} outer brace
        let brace_token = braced!(content in input); // parse the brace
        while !content.is_empty() {
            let stmt: ConfigStmt = content.parse()?;
            contents.push(stmt);
        }
        let eos = if content.peek(Semi) {
            Some(content.parse::<Semi>()?)
        } else {
            None
        };
        Ok(ConfigBlock {
            key,
            brace_token,
            contents,
            eos,
        })
    }
}
