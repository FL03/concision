/*
    Appellation: store <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(any(feature = "alloc", feature = "std"))]
use super::{ParamKind, Parameter};
use ndarray::prelude::{Dimension, Ix2};
use num::Float;

#[cfg(all(feature = "alloc", no_std))]
use alloc::collections::BTreeMap as Map;
#[cfg(feature = "std")]
use std::collections::HashMap as Map;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ParamStore<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    store: Map<ParamKind, Parameter<T, D>>,
}

impl<T, D> ParamStore<T, D>
where
    D: Dimension,
    T: Float,
{
    pub fn new() -> Self {
        Self { store: Map::new() }
    }

    pub fn get(&self, kind: &ParamKind) -> Option<&Parameter<T, D>> {
        self.store.get(kind)
    }

    pub fn get_mut(&mut self, kind: &ParamKind) -> Option<&mut Parameter<T, D>> {
        self.store.get_mut(kind)
    }

    pub fn insert(&mut self, param: Parameter<T, D>) {
        self.store.insert(param.kind().clone(), param);
    }

    pub fn remove(&mut self, kind: &ParamKind) -> Option<Parameter<T, D>> {
        self.store.remove(kind)
    }
}

impl<T, D> Extend<Parameter<T, D>> for ParamStore<T, D>
where
    D: Dimension,
    T: Float,
{
    fn extend<I: IntoIterator<Item = Parameter<T, D>>>(&mut self, iter: I) {
        for param in iter {
            self.insert(param);
        }
    }
}

macro_rules! impl_into_iter {
    ($($p:ident)::*) => {
        impl_into_iter!(@impl $($p)::*);
    };
    (@impl $($p:ident)::*) => {
        impl<T, D> IntoIterator for ParamStore<T, D>
        where
            D: Dimension,
            T: Float,
        {
            type Item = (ParamKind, Parameter<T, D>);
            type IntoIter = $($p)::*::IntoIter<ParamKind, Parameter<T, D>>;

            fn into_iter(self) -> Self::IntoIter {
                self.store.into_iter()
            }
        }

        impl<'a, T, D> IntoIterator for &'a mut ParamStore<T, D>
        where
            D: Dimension,
            T: Float,
        {
            type Item = (&'a ParamKind, &'a mut Parameter<T, D>);
            type IntoIter = $($p)::*::IterMut<'a, ParamKind, Parameter<T, D>>;

            fn into_iter(self) -> Self::IntoIter {
                self.store.iter_mut()
            }
        }
    };

}
#[cfg(feature = "std")]
impl_into_iter!(std::collections::hash_map);
#[cfg(not(feature = "std"))]
impl_into_iter!(alloc::collections::btree_map);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_store() {
        let (inputs, outputs) = (5, 3);

        let _shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let _params = ParamStore::<f64>::new();
    }
}
