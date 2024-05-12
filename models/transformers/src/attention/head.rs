/*
    Appellation: head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::ParamsBase;
use nd::*;

pub struct AttentionHead<A = f64, S = OwnedRepr<A>, D = Ix2>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    params: ParamsBase<S, D>,
}

impl<A, S, D> AttentionHead<A, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn from_params(params: ParamsBase<S, D>) -> Self {
        Self { params }
    }

    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_params(ParamsBase::builder(shape, builder))
    }

    pub fn from_elem<Sh>(shape: Sh, value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
    {
        Self::from_params(ParamsBase::from_elem(shape, value))
    }
    /// Returns a reference to the underlying parameters.
    pub fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }
    /// Returns a mutable reference to the underlying parameters.
    pub fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }

    access!(params::<q, k, v>);
    ndbuilder!(new::default() where A: Default, S: DataOwned);
    ndbuilder!(ones() where A: Clone + num::One, S: DataOwned);
    ndbuilder!(zeros() where A: Clone + num::Zero, S: DataOwned);
}
