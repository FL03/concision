/*
    Appellation: qkv <module>
    Contrib: @FL03
*/
use cnc::Forward;
use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Data, DataOwned, Dimension, Ix2, RawData, ShapeBuilder};
use num_traits::{One, Zero};

pub type Qkv<A = f64, D = Ix2> = QkvParamsBase<ndarray::OwnedRepr<A>, D>;

/// This object is designed to store the parameters of the QKV (Query, Key, Value)
pub struct QkvParamsBase<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) query: ArrayBase<S, D>,
    pub(crate) key: ArrayBase<S, D>,
    pub(crate) value: ArrayBase<S, D>,
}

impl<A, S, D> QkvParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn new(query: ArrayBase<S, D>, key: ArrayBase<S, D>, value: ArrayBase<S, D>) -> Self {
        Self { query, key, value }
    }
    pub fn from_elem<Sh: ShapeBuilder<Dim = D>>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        S: DataOwned,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let query = ArrayBase::from_elem(dim.clone(), elem.clone());
        let key = ArrayBase::from_elem(dim.clone(), elem.clone());
        let value = ArrayBase::from_elem(dim.clone(), elem);
        Self::new(query, key, value)
    }

    pub fn default<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        Self::from_elem(shape, A::default())
    }

    pub fn ones<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self
    where
        A: Clone + One,
        S: DataOwned,
    {
        Self::from_elem(shape, A::one())
    }

    pub fn zeros<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        S: DataOwned,
    {
        Self::from_elem(shape, A::zero())
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

    pub fn set_key(&mut self, key: ArrayBase<S, D>) -> &mut Self {
        *self.key_mut() = key;
        self
    }

    pub fn set_query(&mut self, query: ArrayBase<S, D>) -> &mut Self {
        *self.query_mut() = query;
        self
    }

    pub fn set_value(&mut self, value: ArrayBase<S, D>) -> &mut Self {
        *self.value_mut() = value;
        self
    }

    pub fn with_key(self, key: ArrayBase<S, D>) -> Self {
        Self { key, ..self }
    }

    pub fn with_query(self, query: ArrayBase<S, D>) -> Self {
        Self { query, ..self }
    }

    pub fn with_value(self, value: ArrayBase<S, D>) -> Self {
        Self { value, ..self }
    }
}

/// This trait is used to implement the forward pass for the QKV parameters.
impl<X, Z, A, S, D> Forward<X> for QkvParamsBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    X: Dot<ArrayBase<S, D>, Output = Z>,
    Z: core::ops::Add<Output = Z>,
    for<'a> Z: core::ops::Add<&'a Z, Output = Z>,
{
    type Output = Z;

    fn forward(&self, input: &X) -> Option<Self::Output> {
        let query = input.dot(&self.query);
        let key = input.dot(&self.key);
        let value = input.dot(&self.value);
        let output = query + key + value;
        Some(output)
    }
}
