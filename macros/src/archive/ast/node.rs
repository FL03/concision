/*
    appellation: node <ast>
    authors: @FL03
*/
use super::{VertexAst, WeightAst};
use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, braced, token};

/// The [`NodeStmt`] struct represents the abstract syntax tree (AST) for individual statements
/// used to define the nodes of a hypergraph.
///
/// ```ignore
/// node: {
///     let v0;
///     let v1 = 50;
/// };
/// ```
///
/// Each node statement consists of:
///
/// - `attrs`: A vector of attributes that can be applied to the node.
/// - `key`: A vertex definition that specifies the identifier of the node.
/// - `value`: An optional weight associated with the node, which can be used to assign a value to the node.
/// - `semi`: A semicolon token that marks the end of the node statement.
#[allow(dead_code)]
pub struct NodeStmt {
    pub attrs: Vec<Attribute>,
    /// the vertex definition of the node;
    pub key: VertexAst,
    /// an optional weight associated with the node
    pub value: Option<WeightAst>,
    /// the end of the node statement definition    
    pub semi: token::Semi,
}

/// The [`NodeBlock`] struct represents the abstract syntax tree (AST) for the block-styled
/// statement defining nodes in a hypergraph.
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
///
/// Each node statement consists of:
///
/// - `attrs`: A vector of attributes that can be applied to the node.
/// - `key`: A vertex definition that specifies the identifier of the node.
/// - `value`: An optional weight associated with the node, which can be used to assign a value to the node.
/// - `semi`: A semicolon token that marks the end of the node statement.
#[allow(dead_code)]
pub struct NodeBlock {
    pub attrs: Vec<Attribute>,
    pub nodes_token: kw::nodes,
    pub brace_tkn: token::Brace,
    pub contents: Punctuated<NodeStmt, token::Semi>,
    pub semi: token::Semi,
}

impl Parse for NodeStmt {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // parse the outer attributes
        let attrs = input.call(syn::Attribute::parse_outer)?;
        // parse the vertex definition
        let vertex = input.parse::<VertexAst>()?;
        // determine if there is a weight associated with the node
        let value = if input.peek(token::Eq) {
            Some(input.parse::<WeightAst>()?)
        } else {
            None
        };
        // end-of-statement token
        let semi = input.parse::<token::Semi>()?;
        // return the parsed node AST
        Ok(Self {
            attrs,
            key: vertex,
            value,
            semi,
        })
    }
}

impl Parse for NodeBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(syn::Attribute::parse_outer)?;
        let nodes_token = input.parse::<kw::nodes>()?;
        let content;
        // parse the brace token
        let brace_tkn = braced!(content in input);
        // parse the block contents into individual node statements
        let contents = content.parse_terminated(NodeStmt::parse, token::Semi)?;
        // parse the end-of-block semicolon
        let semi = input.parse::<token::Semi>()?;
        Ok(Self {
            attrs,
            nodes_token,
            brace_tkn,
            contents,
            semi,
        })
    }
}
