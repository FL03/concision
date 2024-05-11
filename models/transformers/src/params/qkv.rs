/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

use num::traits::{One, Zero};

pub struct QKVBase<S = OwnedRepr<f64>, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) q: ArrayBase<S, D>,
    pub(crate) k: ArrayBase<S, D>,
    pub(crate) v: ArrayBase<S, D>,
}

impl<A, S, D> QKVBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        Self {
            q: builder(dim.clone()),
            k: builder(dim.clone()),
            v: builder(dim),
        }
    }

    access!(q, k, v);

    qkv_builder!(new.default where A: Default, S: DataOwned);
    qkv_builder!(ones.ones where A: Clone + One, S: DataOwned);
    qkv_builder!(zeros.zeros where A: Clone + Zero, S: DataOwned);

    pub fn as_views(&self) -> (ArrayView<A, D>, ArrayView<A, D>, ArrayView<A, D>)
    where
        S: Data,
    {
        (self.q.view(), self.k.view(), self.v.view())
    }

    /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
    pub fn dim(&self) -> D::Pattern {
        self.q.dim()
    }
    /// Get the rank of the parameters; i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        self.q.ndim()
    }
    /// Returns the raw dimension ([D](ndarray::Dimension)) of the parameters.
    pub fn raw_dim(&self) -> D {
        self.q.raw_dim()
    }

    pub fn shape(&self) -> &[usize] {
        self.q.shape()
    }

    param_views!(to_owned::<OwnedRepr>(&self) where A: Clone, S: Data);
    param_views!(to_shared::<OwnedArcRepr>(&self) where A: Clone, S: DataShared);
    param_views!(view::<'a, ViewRepr>(&self) where S: Data);
}
