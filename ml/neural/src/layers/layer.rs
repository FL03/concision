/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neurons::Node;

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}