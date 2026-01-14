/*
    Appellation: impl_params_serde <module>
    Created At: 2026.01.13:18:35:20
    Contrib: @FL03
*/
#![cfg(feature = "serde")]
use crate::params_base::ParamsBase;
use ndarray::{Data, DataOwned, Dimension, RawData};
use serde::de::{Deserialize, Deserializer, Error, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use core::marker::PhantomData;

const FIELDS: [&str; 2] = ["bias", "weights"];

struct ParamsBaseVisitor<S, D>
where
    D: Dimension,
    S: RawData,
{
    marker: PhantomData<(S, D)>,
}

impl<'a, A, S, D> Visitor<'a> for ParamsBaseVisitor<S, D>
where
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
    A: Deserialize<'a>,
    <D as ndarray::Dimension>::Smaller: Deserialize<'a>,
{
    type Value = ParamsBase<S, D, A>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a ParamsBase object")
    }

    fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
    where
        V: serde::de::SeqAccess<'a>,
    {
        let bias = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;
        let weights = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(2, &self))?;

        Ok(ParamsBase { bias, weights })
    }
}

impl<'a, A, S, D> Deserialize<'a> for ParamsBase<S, D, A>
where
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
    A: Deserialize<'a>,
    <D as ndarray::Dimension>::Smaller: Deserialize<'a>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'a>,
    {
        deserializer.deserialize_struct(
            "ParamsBase",
            &FIELDS,
            ParamsBaseVisitor {
                marker: PhantomData,
            },
        )
    }
}

impl<A, S, D> Serialize for ParamsBase<S, D, A>
where
    A: Serialize,
    D: Dimension + Serialize,
    S: Data<Elem = A>,
    <D as ndarray::Dimension>::Smaller: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut state = serializer.serialize_struct("ParamsBase", 2)?;
        state.serialize_field("bias", self.bias())?;
        state.serialize_field("weights", self.weights())?;
        state.end()
    }
}
