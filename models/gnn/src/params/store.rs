/*
    Appellation: store <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use petgraph::graph::Graph;
use petgraph::{Directed, Direction};

pub struct GraphStore<K, V, Q = Directed>
where
    Q: Direction,
{
    pub(crate) params: Graph<K, V, Q>,
}

impl<K, V, Q> GraphStore<K, V, Q> {
    pub fn new(params: Graph<K, V, Q>) -> Self {
        Self { params }
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.params.node_weight(key)
    }
}
