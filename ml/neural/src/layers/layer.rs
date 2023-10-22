/*
    Appellation: layer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::LayerType;
use crate::neurons::activate::Activator;
use crate::neurons::Node;
use ndarray::Array2;
use std::ops;

pub trait L<T>
where
    T: num::Float + 'static,
{
    type Activator: Activator<T>;

    fn bias(&self) -> &Array2<T>;

    fn weights(&self) -> &Array2<T>;

    fn activator(&self) -> &Self::Activator;

    fn activate(&self, args: &Array2<T>) -> Array2<T> {
        let z = args.dot(self.weights()) + self.bias();
        z.mapv(|x| self.activator().activate(x))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layer {
    layer: LayerType,
    nodes: Vec<Node>,
}

impl Layer {
    pub fn new(layer: LayerType, nodes: Vec<Node>) -> Self {
        Self { layer, nodes }
    }

    pub fn layer(&self) -> &LayerType {
        &self.layer
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn set_layer(&mut self, layer: LayerType) {
        self.layer = layer;
    }
}

impl AsRef<[Node]> for Layer {
    fn as_ref(&self) -> &[Node] {
        &self.nodes
    }
}

impl AsMut<[Node]> for Layer {
    fn as_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }
}

impl IntoIterator for Layer {
    type Item = Node;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
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
