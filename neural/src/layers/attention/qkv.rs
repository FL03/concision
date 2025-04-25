/*
    Appellation: qkv <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, DataOwned, Dimension, Ix2, RawData, ShapeBuilder};

/// This object is designed to store the parameters of the QKV (Query, Key, Value)
pub struct QkvParams<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) query: ArrayBase<S, D>,
    pub(crate) key: ArrayBase<S, D>,
    pub(crate) value: ArrayBase<S, D>,
}

impl<A, S, D> QkvParams<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn new(query: ArrayBase<S, D>, key: ArrayBase<S, D>, value: ArrayBase<S, D>) -> Self {
        Self { query, key, value }
    }

    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + num_traits::One,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let query = ArrayBase::ones(dim.clone());
        let key = ArrayBase::ones(dim.clone());
        let value = ArrayBase::ones(dim);
        Self { query, key, value }
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + num_traits::Zero,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let query = ArrayBase::zeros(dim.clone());
        let key = ArrayBase::zeros(dim.clone());
        let value = ArrayBase::zeros(dim);
        Self { query, key, value }
    }
    /// returns an immutable reference to the key parameters
    pub const fn key(&self) -> &ArrayBase<S, D> {
        &self.key
    }
    /// returns a mutable reference to the key parameters
    pub fn key_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.key
    }
    /// returns an immutable reference to the query parameters
    pub const fn query(&self) -> &ArrayBase<S, D> {
        &self.query
    }
    /// returns a mutable reference to the query parameters
    pub fn query_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.query
    }
    /// returns an immutable reference to the value parameters
    pub const fn value(&self) -> &ArrayBase<S, D> {
        &self.value
    }
    /// returns a mutable reference to the value parameters
    pub fn value_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.value
    }
}
