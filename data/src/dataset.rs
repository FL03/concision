/*
   Appellation: dataset <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub mod group;

/// A dataset is a collection of records, targets, and weights.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize), serde(rename_all = "lowercase"))]
pub struct Dataset<D, T, W> {
    pub records: D,
    pub targets: T,
    pub weights: W,
}

impl<D, T, W> Dataset<D, T, W> {
    pub fn new(records: D, targets: T, weights: W) -> Self {
        Self {
            records,
            targets,
            weights,
        }
    }

    pub fn records(&self) -> &D {
        &self.records
    }

    pub fn targets(&self) -> &T {
        &self.targets
    }

    pub fn weights(&self) -> &W {
        &self.weights
    }
}

impl<D, T, W> core::fmt::Display for Dataset<D, T, W>
where
    D: core::fmt::Display,
    T: core::fmt::Display,
    W: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{{ records: {}, targets: {}, weights: {} }}", self.records, self.targets, self.weights)
    }
}
