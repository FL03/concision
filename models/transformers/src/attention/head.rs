/*
    Appellation: head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::QKVBase;
use nd::*;

pub struct AttentionHead<A = f64, S = OwnedRepr<A>, D = Ix2>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    params: QKVBase<S, D>,
}

impl<A, S, D> AttentionHead<A, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn from_params(params: QKVBase<S, D>) -> Self {
        Self { params }
    }

    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_params(QKVBase::builder(shape, builder))
    }

    pub fn params(&self) -> &QKVBase<S, D> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut QKVBase<S, D> {
        &mut self.params
    }

    access!(params::<q, k, v>);
    fwd_builder!(new.default where A: Default, S: DataOwned);
    fwd_builder!(ones.ones where A: Clone + num::One, S: DataOwned);
    fwd_builder!(zeros.zeros where A: Clone + num::Zero, S: DataOwned);
}
