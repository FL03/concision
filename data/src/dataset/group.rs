/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataGroup<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    data: Array<T, D>,
    targets: Array<T, D>,
}

impl<T, D> DataGroup<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn new(data: Array<T, D>, targets: Array<T, D>) -> Self {
        Self { data, targets }
    }

    pub fn zeros(ds: impl IntoDimension<Dim = D>, ts: impl IntoDimension<Dim = D>) -> Self {
        Self::new(Array::zeros(ds), Array::zeros(ts))
    }

    pub fn inputs(&self) -> usize {
        self.data.shape().last().unwrap().clone()
    }

    pub fn samples(&self) -> usize {
        self.data.shape().first().unwrap().clone()
    }

    pub fn data(&self) -> &Array<T, D> {
        &self.data
    }

    pub fn targets(&self) -> &Array<T, D> {
        &self.targets
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Serialize};

    impl<'a, T, D> Deserialize<'a> for DataGroup<T, D>
    where
        T: Deserialize<'a> + Float,
        D: Deserialize<'a> + Dimension,
    {
        fn deserialize<Der>(deserializer: Der) -> Result<Self, Der::Error>
        where
            Der: serde::Deserializer<'a>,
        {
            let (data, targets) = Deserialize::deserialize(deserializer)?;
            Ok(Self::new(data, targets))
        }
    }

    impl<T, D> Serialize for DataGroup<T, D>
    where
        T: Float + Serialize,
        D: Dimension + Serialize,
    {
        fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
        where
            Ser: serde::Serializer,
        {
            (self.data(), self.targets()).serialize(serializer)
        }
    }
}
