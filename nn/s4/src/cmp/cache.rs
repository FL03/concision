/*
    Appellation: cache <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::Store;
use ndarray::prelude::{Array, Dimension, Ix2};
// use num::{Complex, Float};
use std::collections::HashMap;

pub struct Cache<T = f64, D = Ix2>
where
    D: Dimension,
{
    cache: HashMap<String, Array<T, D>>,
}

impl<T, D> Cache<T, D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
}

impl<T, D> Store<String, Array<T, D>> for Cache<T, D>
where
    D: Dimension,
{
    fn get(&self, key: &String) -> Option<&Array<T, D>> {
        self.cache.get(key)
    }

    fn get_mut(&mut self, key: &String) -> Option<&mut Array<T, D>> {
        self.cache.get_mut(key)
    }

    fn insert(&mut self, key: String, value: Array<T, D>) -> Option<Array<T, D>> {
        self.cache.insert(key, value)
    }

    fn remove(&mut self, key: &String) -> Option<Array<T, D>> {
        self.cache.remove(key)
    }
}

impl<T, D> Extend<(String, Array<T, D>)> for Cache<T, D>
where
    D: Dimension,
{
    fn extend<I: IntoIterator<Item = (String, Array<T, D>)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<T, D> FromIterator<(String, Array<T, D>)> for Cache<T, D>
where
    D: Dimension,
{
    fn from_iter<I: IntoIterator<Item = (String, Array<T, D>)>>(iter: I) -> Self {
        let mut cache = Self::new();
        for (key, value) in iter {
            cache.insert(key, value);
        }
        cache
    }
}

impl<T, D> IntoIterator for Cache<T, D>
where
    D: Dimension,
{
    type Item = (String, Array<T, D>);
    type IntoIter = std::collections::hash_map::IntoIter<String, Array<T, D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.cache.into_iter()
    }
}

impl<'a, T, D> IntoIterator for &'a mut Cache<T, D>
where
    D: Dimension,
{
    type Item = (&'a String, &'a mut Array<T, D>);
    type IntoIter = std::collections::hash_map::IterMut<'a, String, Array<T, D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.cache.iter_mut()
    }
}
