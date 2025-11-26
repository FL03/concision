/*
    appellation: impl_model_params_serde <module>
    authors: @FL03
*/
use crate::ModelParamsBase;

use crate::RawHidden;
use core::marker::PhantomData;
use ndarray::{Data, DataOwned, Dimension, RawData};
use serde::de::{Deserialize, Deserializer, Error, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

/// a constant defining the various fields of the [`ModelParamsBase`] type that are used for
/// serialization and deserialization.
const FIELDS: [&str; 3] = ["input", "hidden", "output"];

struct ModelParamsBaseVisitor<S, D, H, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: RawHidden<S, D>,
{
    marker: PhantomData<(S, D, H, A)>,
}

impl<'a, A, S, D, H> Visitor<'a> for ModelParamsBaseVisitor<S, D, H, A>
where
    A: Deserialize<'a>,
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
    H: RawHidden<S, D> + Deserialize<'a>,
    <D as Dimension>::Smaller: Deserialize<'a>,
{
    type Value = ModelParamsBase<S, D, H, A>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("The visitor is expecting to receive a `ModelParamsBase` object.")
    }

    fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
    where
        V: serde::de::SeqAccess<'a>,
    {
        let input = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;
        let hidden = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(2, &self))?;
        let output = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(3, &self))?;

        Ok(ModelParamsBase {
            input,
            hidden,
            output,
        })
    }
}

impl<'a, A, S, D, H> Deserialize<'a> for ModelParamsBase<S, D, H, A>
where
    A: Deserialize<'a>,
    D: Dimension + Deserialize<'a>,
    S: DataOwned<Elem = A>,
    H: RawHidden<S, D> + Deserialize<'a>,
    <D as Dimension>::Smaller: Deserialize<'a>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'a>,
    {
        deserializer.deserialize_struct(
            "ModelParamsBase",
            &FIELDS,
            ModelParamsBaseVisitor {
                marker: PhantomData,
            },
        )
    }
}

impl<A, S, D, H> Serialize for ModelParamsBase<S, D, H, A>
where
    A: Serialize,
    D: Dimension + Serialize,
    S: Data<Elem = A>,
    H: RawHidden<S, D> + Serialize,
    <D as Dimension>::Smaller: Serialize,
{
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        let mut state = serializer.serialize_struct("ModelParamsBase", 3)?;
        state.serialize_field("input", &self.input)?;
        state.serialize_field("hidden", &self.hidden)?;
        state.serialize_field("output", &self.output)?;
        state.end()
    }
}
