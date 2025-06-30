/*
    appellation: impl_tensor_serde <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use core::marker::PhantomData;
use ndarray::{Data, DataOwned, Dimension, RawData};
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

/// a constant defining the fields of the `TensorBase` struct for serialization
const FIELDS: [&str; 1] = ["store"];

pub struct TensorBaseVisitor<S, D>
where
    D: Dimension,
    S: RawData,
{
    _phantom: PhantomData<(S, D)>,
}

impl<'a, A, S, D> Visitor<'a> for TensorBaseVisitor<S, D>
where
    A: Deserialize<'a>,
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
{
    type Value = TensorBase<S, D>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a tensor with data")
    }

    fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
    where
        V: serde::de::SeqAccess<'a>,
    {
        let store = seq
            .next_element()?
            .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
        Ok(TensorBase { store })
    }
}

impl<'a, A, S, D> Deserialize<'a> for TensorBase<S, D>
where
    A: Deserialize<'a>,
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'a>,
    {
        deserializer.deserialize_struct(
            "TensorBase",
            &FIELDS,
            TensorBaseVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

impl<A, S, D> Serialize for TensorBase<S, D>
where
    A: Serialize,
    D: Dimension + Serialize,
    S: Data<Elem = A>,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut state = serializer.serialize_struct("TensorBase", 1)?;
        state.serialize_field("data", self.store())?;
        state.end()
    }
}
