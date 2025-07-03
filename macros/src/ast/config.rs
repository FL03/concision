/*
    appellation: layer <module>
    authors: @FL03
*/
use crate::ast::BlockHeader;
use crate::kw;

use syn::parse::{Parse, ParseStream};
use syn::token::{Brace, Colon, Eq, Semi};
use syn::{Expr, Ident, Type, braced};

#[allow(dead_code)]
pub struct ConfigAst {
    // the `config` keyword
    pub header: BlockHeader<kw::config>,
    pub brace_token: Brace,
    pub params: Option<ParamConfigBlock>,
    pub eos: Option<Semi>,
}

/// The [`ParamConfigBlock`] struct represents a block of layer definitions in the model DSL.
///
/// ```no_run
/// model_config! {
///     config: MyConfig {
///         params: MyParams {
///             let x: f32; // default value is `0.0`
///             let y: i32 = 42;
///             let z: String = "Hello, world!".to_string();
///         };
///     }
/// }
/// ```
#[allow(dead_code)]
pub struct ParamConfigBlock {
    pub key: BlockHeader<kw::hyperparams>,
    pub brace_token: Brace,
    pub contents: Vec<ParamConfigStmt>,
    pub eos: Option<Semi>,
}
/// The [`ParamConfigStmt`] defines the structure of a single layer statement within a [`ParamConfigBlock`]
///
/// ```no_run
/// $key: $ty = $default;
/// ```
#[allow(dead_code)]
pub struct ParamConfigStmt {
    pub key: Ident,
    pub colon_token: Colon,
    pub typed: Type,
    /// the layer expression, which can be an identifier, an expression, or a raw token stream
    pub value: Option<ValueExpr>,
    /// the end-of-statement token
    pub eos: Semi,
}
#[allow(dead_code)]
pub struct ValueExpr {
    pub eq_token: Eq,
    /// the layer expression, which can be an identifier, an expression, or a raw token stream
    pub value: Expr,
}

impl Parse for ValueExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the equals token
        let eq_token = input.parse::<Eq>()?;
        // parse the expression
        let value = input.parse::<Expr>()?;
        // create the ValueExpr instance
        Ok(ValueExpr { eq_token, value })
    }
}

impl Parse for ParamConfigStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the identifier
        let key = input.parse::<Ident>()?;
        // parse the domain expression
        let colon_token = input.parse::<Colon>()?;
        // parse the type of the hyperparam
        let typed = input.parse::<Type>()?;
        //
        let value = input.parse::<ValueExpr>().ok();
        let eos = input.parse::<Semi>()?;

        // create the ParamConfigStmt instance
        Ok(ParamConfigStmt {
            key,
            colon_token,
            typed,
            value,
            eos,
        })
    }
}

impl Parse for ParamConfigBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `layer` keyword
        let key = input.parse::<BlockHeader<kw::hyperparams>>()?;
        // create a store for the layer block
        let mut contents = Vec::new();
        // create a buffer for the braced content
        let content; // the layer {} outer brace
        let brace_token = braced!(content in input); // parse the brace
        while !content.is_empty() {
            let stmt: ParamConfigStmt = content.parse()?;
            contents.push(stmt);
        }
        let eos = if content.peek(Semi) {
            Some(content.parse::<Semi>()?)
        } else {
            None
        };
        Ok(ParamConfigBlock {
            key,
            brace_token,
            contents,
            eos,
        })
    }
}

impl Parse for ConfigAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the `config` keyword
        let header = input.parse::<BlockHeader<kw::config>>()?;
        // create a store for the layer block
        let mut params = None;
        // create a buffer for the braced content
        let content; // the layer {} outer brace
        // use the macro to parse the brace and its contents
        let brace_token = braced!(content in input);
        // parse the contents of the block
        while !content.is_empty() {
            // handle the hyperparameters block
            if content.peek(kw::hyperparams) {
                params = Some(content.parse::<ParamConfigBlock>()?);
            } else {
                // if we encounter an unexpected token, return an error
                return Err(syn::Error::new(
                    content.span(),
                    "expected `hyperparams` block",
                ));
            }
        }
        // optionally, parse the end-of-statement token
        let eos = if content.peek(Semi) {
            Some(content.parse::<Semi>()?)
        } else {
            None
        };
        // create the ConfigAst instance
        Ok(ConfigAst {
            header,
            brace_token,
            params,
            eos,
        })
    }
}
