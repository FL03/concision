/*
   Appellation: store <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Store
pub use self::storage::*;

pub(crate) mod storage;

pub trait Store<K, V> {
    fn get(&self, key: &K) -> Option<&V>;

    fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    fn insert(&mut self, key: K, value: V) -> Option<V>;

    fn remove(&mut self, key: &K) -> Option<V>;
}

#[cfg(test)]
mod tests {}
