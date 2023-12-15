/*
    Appellation: store <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{ParamKind, Parameter};
use ndarray::prelude::{Dimension, Ix2};
use num::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParamStore<T = f64, D = Ix2> where T: Float, D: Dimension {
    store: HashMap<ParamKind, Parameter<T, D>>,
}

impl<T, D> ParamStore<T, D> where D: Dimension, T: Float, {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn insert(&mut self, param: Parameter<T, D>) {
        self.store.insert(param.kind().clone(), param);
    }

}


impl<T, D> Extend<Parameter<T, D>> for ParamStore<T, D> where D: Dimension, T: Float, {
    fn extend<I: IntoIterator<Item = Parameter<T, D>>>(&mut self, iter: I) {
        for param in iter {
            self.insert(param);
        }
    }
}

impl<T, D> IntoIterator for ParamStore<T, D> where D: Dimension, T: Float, {
    type Item = (ParamKind, Parameter<T, D>);
    type IntoIter = std::collections::hash_map::IntoIter<ParamKind, Parameter<T, D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_store() {
        let (inputs, outputs) = (5, 3);

        let shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let params = ParamStore::<f64>::new();

    }
}
