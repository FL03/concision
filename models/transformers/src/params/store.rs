/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

use num::traits::{One, Zero};

pub struct ParamsBase<S = OwnedRepr<f64>, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) q: ArrayBase<S, D>,
    pub(crate) k: ArrayBase<S, D>,
    pub(crate) v: ArrayBase<S, D>,
}

impl<A, S, D> ParamsBase<S, D>
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

    ndbuilder!(new::default() where A: Default, S: DataOwned);
    ndbuilder!(ones() where A: Clone + One, S: DataOwned);
    ndbuilder!(zeros() where A: Clone + Zero, S: DataOwned);

    concision::getters!(q, k, v => ArrayBase<S, D>);

    pub fn from_elem<Sh>(shape: Sh, value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
    {
        let dim = shape.into_shape().raw_dim().clone();
        Self {
            q: ArrayBase::from_elem(dim.clone(), value.clone()),
            k: ArrayBase::from_elem(dim.clone(), value.clone()),
            v: ArrayBase::from_elem(dim, value),
        }
    }

    pub fn as_qkv(&self) -> (ArrayView<A, D>, ArrayView<A, D>, ArrayView<A, D>)
    where
        S: Data,
    {
        (self.q.view(), self.k.view(), self.v.view())
    }
    /// Consumes the current parameters, returning a three-tuple the Q, K, and V matrices respectivley.
    pub fn into_qkv(self) -> (ArrayBase<S, D>, ArrayBase<S, D>, ArrayBase<S, D>)
    where
        S: DataOwned,
    {
        (self.q, self.k, self.v)
    }
    /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
    pub fn dim(&self) -> D::Pattern {
        self.q().dim()
    }
    /// Get the rank of the parameters; i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        self.q().ndim()
    }
    /// Returns the raw dimension ([D](ndarray::Dimension)) of the parameters.
    pub fn raw_dim(&self) -> D {
        self.q().raw_dim()
    }
    /// Returns a slice of the current shape of the parameters.
    pub fn shape(&self) -> &[usize] {
        self.q().shape()
    }
    ndview!(into_owned::<OwnedRepr>(self) where A: Clone, S: Data);
    ndview!(to_owned::<OwnedRepr>(&self) where A: Clone, S: Data);

    ndview!(into_shared::<OwnedArcRepr>(self) where A: Clone, S: DataOwned);
    ndview!(to_shared::<OwnedArcRepr>(&self) where A: Clone, S: DataShared);

    ndview!(view::<'a, ViewRepr>(&self) where S: Data);
    ndview!(view_mut::<'a, ViewRepr>(&mut self) where S: DataMut);
}
