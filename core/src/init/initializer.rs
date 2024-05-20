/*
    Appellation: initializer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Initialize;
use core::marker::PhantomData;
use nd::prelude::*;
use nd::DataOwned;
use rand_distr::{Distribution, StandardNormal};

pub struct InitializerBase<A = f64, D = Ix2, Dst = StandardNormal> where D: Dimension, Dst: Clone + Distribution<A> {
    pub(crate) dim: D,
    pub(crate) distr: Dst,
    pub(crate) _dtype: PhantomData<A>,
}

impl<A, D, Dst> InitializerBase<A, D, Dst> where D: Dimension, Dst: Clone + Distribution<A> {
    pub fn new(dim: D, distr: Dst) -> Self {
        Self { dim, distr, _dtype: PhantomData::<A> }
    }

    pub fn init<S>(self) -> ArrayBase<S, D> where S: DataOwned<Elem = A> {
        ArrayBase::rand(self.dim, self.distr)
    }
}
