/*
   Appellation: dataset <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Dataset<D, T, W> {
    pub data: D,
    pub targets: T,
    pub weights: W,
}

impl<D, T, W> Dataset<D, T, W> {
    pub fn new(data: D, targets: T, weights: W) -> Self {
        Self {
            data,
            targets,
            weights,
        }
    }

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn targets(&self) -> &T {
        &self.targets
    }

    pub fn weights(&self) -> &W {
        &self.weights
    }
}

impl<D, T, W> std::fmt::Display for Dataset<D, T, W>
where
    D: Serialize,
    T: Serialize,
    W: Serialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
