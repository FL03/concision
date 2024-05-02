/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::*;
use strum::{AsRefStr, EnumDiscriminants, EnumIs, VariantNames};

#[derive(AsRefStr, EnumDiscriminants, EnumIs, VariantNames)]
#[cfg_attr(
    feature = "serde",
    strum_discriminants(
        derive(serde::Deserialize, serde::Serialize),
        serde(rename_all = "lowercase", untagged),
    )
)]
#[non_exhaustive]
#[strum(serialize_all = "lowercase")]
#[strum_discriminants(
    name(Param),
    derive(
        AsRefStr,
        Hash,
        Ord,
        PartialOrd,
        VariantNames,
        strum::Display,
        strum::EnumCount,
        EnumIs,
        strum::EnumIter,
        strum::EnumString,
        strum::VariantArray
    ),
    strum(serialize_all = "lowercase")
)]
pub enum Entry<S, D>
where
    S: RawData,
    D: RemoveAxis,
{
    Bias(ArrayBase<S, D::Smaller>),
    Weight(ArrayBase<S, D>),
}

impl<A, S, D> Entry<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    pub fn bias(data: ArrayBase<S, D::Smaller>) -> Self {
        Self::Bias(data)
    }

    pub fn weight(data: ArrayBase<S, D>) -> Self {
        Self::Weight(data)
    }
}

impl Param {
    pub fn bias() -> Self {
        Self::Bias
    }

    pub fn weight() -> Self {
        Self::Weight
    }
}