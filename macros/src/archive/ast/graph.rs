/*
    appellation: graph <ast>
    authors: @FL03
*/
use crate::ast::{EdgeBlock, NodeBlock};
use crate::kw;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, braced, token};

#[allow(dead_code)]
/// The [`GraphAst`] struct represents the abstract syntax tree (AST) for a hypergraph macro.
/// Overall, it is supposed to feel similar to
///
/// ```ignore
/// hypergraph! {
///     graph {
///         nodes: {
///             let v0;
///             let v1 = 90;
///             let v2 = 100;
///         };
///         edges: {
///             let e0: [v0, v1];
///             let e1: [v0, v1, v2] = 50;
///         };
///     }
/// }
/// ```
pub struct GraphAst {
    pub graph: Ident,
    pub brace_token: token::Brace,
    pub node_block: Option<NodeBlock>,
    pub edge_block: Option<EdgeBlock>,
    pub semi_token: Option<token::Semi>,
}

impl Parse for GraphAst {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let graph: Ident = input.parse()?;
        let content; // the graph {} outer brace
        let brace_token = braced!(content in input);

        let mut node_block: Option<NodeBlock> = None;
        let mut edge_block: Option<EdgeBlock> = None;
        
        while !content.is_empty() {
            // if present, parse the edge block
            if content.peek(kw::edges) {
                // parse the edges block
                edge_block = content.parse().ok();
            } else if content.peek(kw::nodes) {
                // parse the nodes block
                node_block = content.parse().ok();
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
        if edge_block.is_some() && node_block.is_none() {
            return Err(syn::Error::new_spanned(
                graph,
                "To define any edges with the macro, you must also define nodes.",
            ));
        }
        // return the parsed graph AST
        Ok(GraphAst {
            graph,
            brace_token,
            node_block,
            edge_block,
            semi_token,
        })
    }
}
