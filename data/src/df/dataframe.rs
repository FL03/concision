/*
   Appellation: dataframe <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, Dimension};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct DataFrame<D: Dimension, T> {
    data: Array<T, D>,
}

impl<D: Dimension, T: Default> DataFrame<D, T> {
    pub fn new() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<D: Dimension, T> std::fmt::Display for DataFrame<D, T>
where
    D: Serialize,
    T: Serialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
