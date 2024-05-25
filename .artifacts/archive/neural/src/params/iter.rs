/*
   Appellation: iter <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K, V> Entry<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Self { key, value }
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn value(&self) -> &V {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

pub struct IntoIter;

pub struct Iter;

pub struct IterMut;
