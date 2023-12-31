/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array1, Array2};

pub trait Records<T> {
    fn features(&self) -> usize;

    fn samples(&self) -> usize;
}

impl<T> Records<T> for Array1<T> {
    fn features(&self) -> usize {
        self.shape()[1]
    }

    fn samples(&self) -> usize {
        self.shape()[0]
    }
}

impl<T> Records<T> for Array2<T> {
    fn features(&self) -> usize {
        self.shape()[1]
    }

    fn samples(&self) -> usize {
        self.shape()[0]
    }
}

pub trait NdArrayExt<T> {}

pub trait Store<K, V> {
    fn get(&self, key: &K) -> Option<&V>;

    fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    fn insert(&mut self, key: K, value: V) -> Option<V>;

    fn remove(&mut self, key: &K) -> Option<V>;
}
