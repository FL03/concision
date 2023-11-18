/*
   Appellation: dataset <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array2, Ix2};
use ndarray::Dimension;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct DataSet<T = f64, D = Ix2>
where
    D: Dimension,
{
    data: Array2<T>,
    targets: Array<T, D>,
}

impl<T, D> DataSet<T, D>
where
    D: Dimension,
    T: Float,
{
    pub fn new(data: Array2<T>, targets: Array<T, D>) -> Self {
        Self { data, targets }
    }
}

impl<T, D> std::fmt::Display for DataSet<T, D>
where
    D: Dimension + Serialize,
    T: Serialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
