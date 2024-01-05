/*
   Appellation: store <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Store
pub use self::{layout::*, storage::*};

pub(crate) mod layout;
pub(crate) mod storage;

use std::collections::{BTreeMap, HashMap};
use std::ops;

pub trait Store<K, V> {
    fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    fn get(&self, key: &K) -> Option<&V>;

    fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    fn insert(&mut self, key: K, value: V) -> Option<V>;

    fn remove(&mut self, key: &K) -> Option<V>;
}

pub trait StoreExt<K, V>: Store<K, V> + ops::Index<K, Output = V> {}

impl<K, V> Store<K, V> for BTreeMap<K, V>
where
    K: Ord,
{
    fn get(&self, key: &K) -> Option<&V> {
        BTreeMap::get(self, key)
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        BTreeMap::get_mut(self, key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        BTreeMap::insert(self, key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        BTreeMap::remove(self, key)
    }
}

impl<K, V> Store<K, V> for HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        HashMap::get_mut(self, key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        HashMap::insert(self, key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        HashMap::remove(self, key)
    }
}

#[cfg(test)]
mod tests {}
