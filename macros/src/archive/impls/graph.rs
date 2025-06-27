/*
    appellation: node <module>
    authors: @FL03
*/
use crate::ast::{EdgeBlock, EdgeStmt, GraphAst, NodeBlock, NodeStmt, WeightAst};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub fn impl_graph(GraphAst {
    graph,
    node_block,
    edge_block,
    ..
}: GraphAst) -> TokenStream {
    // if there are no nodes or edges, return an empty TokenStream
    if node_block.is_none() && edge_block.is_none() {
        return TokenStream::new();
    }
    // if edges are defined, but no nodes, return an error
    if edge_block.is_some() && node_block.is_none() {
        return syn::Error::new_spanned(
            graph,
            "edges cannot be defined without nodes",
        )
        .to_compile_error()
        .into();
    }
    // initialize a vector to hold the statements
    let mut stmts = Vec::new();
    // handle the node block
    if let Some(NodeBlock { contents, .. }) = &node_block {
        contents.iter().for_each(|node| {
            let stmt = handle_node(&graph, node);
            stmts.push(stmt);
        });
    }
    // handle the edge block
    if let Some(EdgeBlock { contents, .. }) = &edge_block {
        contents.iter().for_each(|edge| {
            let stmt = handle_edge(&graph, edge);
            stmts.push(stmt);
        });
    }
    // generate the output code
    quote! {
        #(#stmts)*
    }
}
/// a method for handling individual edge statements
pub fn handle_edge(g: &Ident, edge: &EdgeStmt) -> proc_macro2::TokenStream {
    let EdgeStmt {
        key, domain, weight, ..
    } = edge;
    if let Some(WeightAst { expr: value, .. }) = weight {
        quote! {
            let #key = #g.add_surface(#domain, #value.into()).expect("failed to add edge");
        }
    } else {
        quote! {
            let #key = #g.add_edge(#domain).expect("failed to add edge");
        }
    }
}

pub fn handle_node(g: &Ident, node: &NodeStmt) -> proc_macro2::TokenStream {
    let NodeStmt { key, value, .. } = node;
    if let Some(WeightAst { expr: value, .. }) = value {
        quote! {
            #key #g.add_node(#value.into()).expect("failed to add node");
        }
    } else {
        quote! {
            #key #g.add_vertex().expect("failed to add node");
        }
    }
}
