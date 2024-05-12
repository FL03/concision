/*
    Appellation: impl_params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::ParamsBase;
use nd::prelude::*;
use nd::{Data, DataOwned, RawDataClone};

impl<S, D> Clone for ParamsBase<S, D>
where
    D: Dimension,
    S: RawDataClone,
{
    fn clone(&self) -> Self {
        Self {
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
        }
    }
}

impl<S, D> Copy for ParamsBase<S, D>
where
    D: Copy + Dimension,
    S: Copy + RawDataClone,
{
}

impl<S, D> Default for ParamsBase<S, D>
where
    D: Dimension,
    S: DataOwned,
    S::Elem: Default,
{
    fn default() -> Self {
        Self {
            q: Default::default(),
            k: Default::default(),
            v: Default::default(),
        }
    }
}

impl<A, S, D> PartialEq for ParamsBase<S, D>
where
    A: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.q == *other.q() && self.k == *other.k() && self.v == *other.v()
    }
}

impl<A, B, S, D, S2, D2> PartialEq<ArrayBase<S2, D2>> for ParamsBase<S, D>
where
    A: PartialEq,
    B: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
    S2: Data<Elem = B>,
    D2: Dimension,
    ArrayBase<S, D>: PartialEq<ArrayBase<S2, D2>>,
{
    fn eq(&self, other: &ArrayBase<S2, D2>) -> bool {
        self.q == *other && self.k == *other && self.v == *other
    }
}
