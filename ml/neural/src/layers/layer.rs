/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neurons::Node;
use std::ops;

#[derive(Clone, Debug, Default, PartialEq)]
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

impl ops::Index<usize> for Layer {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl ops::IndexMut<usize> for Layer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

impl ops::Index<ops::Range<usize>> for Layer {
    type Output = [Node];

    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        &self.nodes[index]
    }
}

impl ops::IndexMut<ops::Range<usize>> for Layer {
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

