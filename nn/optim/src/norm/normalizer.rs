/*
    Appellation: normalizer <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Norm, Norms};
use ndarray::prelude::{Array, NdFloat};
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Normalizer {
    pub mode: Norms,
}

impl Normalizer {
    pub fn new(mode: Norms) -> Self {
        Self { mode }
    }

    pub fn normalize<S, T>(&self, args: &S) -> T
    where
        S: Norm<T>,
    {
        match self.mode {
            Norms::L0 => args.l0(),
            Norms::L1 => args.l1(),
            Norms::L2 => args.l2(),
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
