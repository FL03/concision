/*
    Appellation: serde <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "serde")]

use crate::params::{Entry, LinearParams};
use nd::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<'a, T, D> Deserialize<'a> for LinearParams<T, D>
where
    T: Deserialize<'a>,
    D: Deserialize<'a> + RemoveAxis,
    <D as Dimension>::Smaller: Deserialize<'a> + Dimension,
{
    fn deserialize<Der>(deserializer: Der) -> Result<Self, Der::Error>
    where
        Der: Deserializer<'a>,
    {
        let (bias, features, weights) = Deserialize::deserialize(deserializer)?;
        Ok(Self {
            bias,
            features,
            weights,
        })
    }
}

impl<T, D> Serialize for LinearParams<T, D>
where
    T: Serialize,
    D: RemoveAxis + Serialize,
    <D as Dimension>::Smaller: Dimension + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        (self.bias(), self.features(), self.weights()).serialize(serializer)
    }
}

impl<A, S, D> Serialize for Entry<S, D>
where
    A: Serialize,
    S: Data<Elem = A>,
    D: RemoveAxis + Serialize,
    <D as Dimension>::Smaller: Dimension + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        match self {
            Self::Bias(bias) => bias.serialize(serializer),
            Self::Weight(weight) => weight.serialize(serializer),
        }
    }
}
