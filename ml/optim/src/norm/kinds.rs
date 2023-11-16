/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Norm;
use ndarray::prelude::{Array, NdFloat};
use ndarray::Dimension;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumIs,
    EnumIter,
    EnumString,
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Norms {
    L0 = 0,
    L1 = 1,
    #[default]
    L2 = 2,
}

impl Norms {
    pub fn l0() -> Self {
        Self::L0
    }

    pub fn l1() -> Self {
        Self::L1
    }

    pub fn l2() -> Self {
        Self::L2
    }

    pub fn normalize<S, T>(&self, args: &S) -> T
    where
        S: Norm<T>,
    {
        use Norms::*;

        match *self {
            L0 => args.l0(),
            L1 => args.l1(),
            L2 => args.l2(),
        }
    }

    pub fn norm_and_scale<T, D>(&self, args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: NdFloat,
    {
        args / self.normalize(args)
    }
}
