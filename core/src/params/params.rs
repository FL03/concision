/*
    Appellation: params <module>
    Contrib: @FL03
*/
use ndarray::prelude::*;
use ndarray::{Data, DataMut, DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

/// The [`ParamsBase`] struct is a generic container for a set of weights and biases for a
/// model. The implementation is designed around the [`ArrayBase`] type from the
/// `ndarray` crate, which allows for flexible and efficient storage of multi-dimensional
/// arrays.
pub struct ParamsBase<S, D = ndarray::Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) bias: ArrayBase<S, D::Smaller>,
    pub(crate) weights: ArrayBase<S, D>,
}

impl<A, S, D> ParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new instance of the [`ParamsBase`] with the given bias and weights
    pub const fn new(bias: ArrayBase<S, D::Smaller>, weights: ArrayBase<S, D>) -> Self {
        Self { bias, weights }
    }
    /// create a new instance of the [`ModelParams`] from the given shape and element;
    pub fn from_elems<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let weights = ArrayBase::from_elem(shape, elem.clone());
        let dim = weights.raw_dim();
        let bias = ArrayBase::from_elem(dim.remove_axis(Axis(0)), elem);
        Self::new(bias, weights)
    }
    /// create an instance of the parameters with all values set to the default value
    pub fn default<Sh>(shape: Sh) -> Self
    where
        A: Clone + Default,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elems(shape, A::default())
    }
    /// initialize the parameters with all values set to zero
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + num_traits::One,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elems(shape, A::one())
    }
    /// create an instance of the parameters with all values set to zero
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + num_traits::Zero,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_elems(shape, A::zero())
    }
    /// returns an immutable reference to the bias
    pub const fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        &self.bias
    }
    /// returns a mutable reference to the bias
    pub const fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        &mut self.bias
    }
    /// returns an immutable reference to the weights
    pub const fn weights(&self) -> &ArrayBase<S, D> {
        &self.weights
    }
    /// returns a mutable reference to the weights
    pub const fn weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.weights
    }
    /// assign the bias
    pub fn assign_bias(&mut self, bias: &ArrayBase<S, D::Smaller>) -> &mut Self
    where
        A: Clone,
        S: DataMut,
    {
        self.bias_mut().assign(bias);
        self
    }
    /// assign the weights
    pub fn assign_weights(&mut self, weights: &ArrayBase<S, D>) -> &mut Self
    where
        A: Clone,
        S: DataMut,
    {
        self.weights_mut().assign(weights);
        self
    }
    /// replace the bias and return the previous state; uses [replace](core::mem::replace)
    pub fn replace_bias(&mut self, bias: ArrayBase<S, D::Smaller>) -> ArrayBase<S, D::Smaller> {
        core::mem::replace(&mut self.bias, bias)
    }
    /// replace the weights and return the previous state; uses [replace](core::mem::replace)
    pub fn replace_weights(&mut self, weights: ArrayBase<S, D>) -> ArrayBase<S, D> {
        core::mem::replace(&mut self.weights, weights)
    }
    /// set the bias
    pub fn set_bias(&mut self, bias: ArrayBase<S, D::Smaller>) -> &mut Self {
        *self.bias_mut() = bias;
        self
    }
    /// set the weights
    pub fn set_weights(&mut self, weights: ArrayBase<S, D>) -> &mut Self {
        *self.weights_mut() = weights;
        self
    }
    /// perform a single backpropagation step
    pub fn backward<X, Y, Z>(&mut self, input: &X, grad: &Y, lr: A) -> crate::Result<Z>
    where
        A: Clone,
        S: Data,
        Self: crate::Backward<X, Y, Elem = A, Output = Z>,
    {
        <Self as crate::Backward<X, Y>>::backward(self, input, grad, lr)
    }
    /// forward propagation
    pub fn forward<X, Y>(&self, input: &X) -> crate::Result<Y>
    where
        A: Clone,
        S: Data,
        Self: crate::Forward<X, Output = Y>,
    {
        <Self as crate::Forward<X>>::forward(self, input)
    }
    /// returns the dimensions of the weights
    pub fn dim(&self) -> D::Pattern {
        self.weights().dim()
    }
    /// an iterator of the parameters; the created iterator zips together an axis iterator over
    /// the columns of the weights and an iterator over the bias
    pub fn iter(&self) -> super::iter::Iter<'_, A, D>
    where
        D: RemoveAxis,
        S: Data,
    {
        super::iter::Iter {
            bias: self.bias().iter(),
            weights: self.weights().axis_iter(Axis(1)),
        }
    }
    /// a mutable iterator of the parameters
    pub fn iter_mut(
        &mut self,
    ) -> core::iter::Zip<
        ndarray::iter::AxisIterMut<'_, A, D::Smaller>,
        ndarray::iter::IterMut<'_, A, D::Smaller>,
    >
    where
        D: RemoveAxis,
        S: DataMut,
    {
        self.weights
            .axis_iter_mut(Axis(1))
            .zip(self.bias.iter_mut())
    }
    /// returns an iterator over the bias
    pub fn iter_bias(&self) -> ndarray::iter::Iter<'_, A, D::Smaller>
    where
        S: Data,
    {
        self.bias().iter()
    }
    /// returns a mutable iterator over the bias
    pub fn iter_bias_mut(&mut self) -> ndarray::iter::IterMut<'_, A, D::Smaller>
    where
        S: DataMut,
    {
        self.bias_mut().iter_mut()
    }
    /// returns an iterator over the weights
    pub fn iter_weights(&self) -> ndarray::iter::Iter<'_, A, D>
    where
        S: Data,
    {
        self.weights().iter()
    }
    /// returns a mutable iterator over the weights; see [`iter_mut`](ArrayBase::iter_mut) for more
    pub fn iter_weights_mut(&mut self) -> ndarray::iter::IterMut<'_, A, D>
    where
        S: DataMut,
    {
        self.weights_mut().iter_mut()
    }
    /// returns true if both the weights and bias are empty; uses [`is_empty`](ArrayBase::is_empty)
    pub fn is_empty(&self) -> bool {
        self.is_weights_empty() && self.is_bias_empty()
    }
    /// returns true if the weights are empty
    pub fn is_weights_empty(&self) -> bool {
        self.weights().is_empty()
    }
    /// returns true if the bias is empty
    pub fn is_bias_empty(&self) -> bool {
        self.bias().is_empty()
    }
    /// the total number of elements within the weight tensor
    pub fn count_weight(&self) -> usize {
        self.weights().len()
    }
    /// the total number of elements within the bias tensor
    pub fn count_bias(&self) -> usize {
        self.bias().len()
    }
    /// returns the raw dimensions of the weights;
    pub fn raw_dim(&self) -> D {
        self.weights().raw_dim()
    }
    /// returns the shape of the parameters; uses the shape of the weight tensor
    pub fn shape(&self) -> &[usize] {
        self.weights().shape()
    }
    /// returns the shape of the bias tensor; the shape should be equivalent to that of the
    /// weight tensor minus the "zero-th" axis
    pub fn shape_bias(&self) -> &[usize] {
        self.bias().shape()
    }
    /// returns the total number of parameters within the layer
    pub fn size(&self) -> usize {
        self.weights().len() + self.bias().len()
    }
    /// returns an owned instance of the parameters
    pub fn to_owned(&self) -> ParamsBase<ndarray::OwnedRepr<A>, D>
    where
        A: Clone,
        S: DataOwned,
    {
        ParamsBase::new(self.bias().to_owned(), self.weights().to_owned())
    }
    /// change the shape of the parameters; the shape of the bias parameters is determined by
    /// removing the "zero-th" axis of the given shape
    pub fn to_shape<Sh>(
        &self,
        shape: Sh,
    ) -> crate::Result<ParamsBase<ndarray::CowRepr<'_, A>, Sh::Dim>>
    where
        A: Clone,
        S: DataOwned,
        Sh: ShapeBuilder,
        Sh::Dim: Dimension + RemoveAxis,
    {
        let shape = shape.into_shape_with_order();
        let dim = shape.raw_dim().clone();
        let bias = self.bias().to_shape(dim.remove_axis(Axis(0)))?;
        let weights = self.weights().to_shape(dim)?;
        Ok(ParamsBase::new(bias, weights))
    }
    /// returns a "view" of the parameters; see [view](ArrayBase::view) for more information
    pub fn view(&self) -> ParamsBase<ndarray::ViewRepr<&'_ A>, D>
    where
        S: Data,
    {
        ParamsBase::new(self.bias().view(), self.weights().view())
    }
    /// returns mutable view of the parameters; see [view_mut](ArrayBase::view_mut) for more information
    pub fn view_mut(&mut self) -> ParamsBase<ndarray::ViewRepr<&'_ mut A>, D>
    where
        S: ndarray::DataMut,
    {
        ParamsBase::new(self.bias.view_mut(), self.weights.view_mut())
    }
}

impl<A, S> ParamsBase<S, Ix1>
where
    S: RawData<Elem = A>,
{
    pub fn from_scalar_bias(bias: A, weights: ArrayBase<S, Ix1>) -> Self
    where
        A: Clone,
        S: DataOwned,
    {
        Self {
            bias: ArrayBase::from_elem((), bias),
            weights,
        }
    }

    pub fn nrows(&self) -> usize {
        self.weights.len()
    }
}

impl<A, S> ParamsBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    pub fn ncols(&self) -> usize {
        self.weights.ncols()
    }

    pub fn nrows(&self) -> usize {
        self.weights.nrows()
    }
}

impl<A, S, D> Clone for ParamsBase<S, D>
where
    D: Dimension,
    S: ndarray::RawDataClone<Elem = A>,
    A: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.bias().clone(), self.weights().clone())
    }
}

impl<A, S, D> Copy for ParamsBase<S, D>
where
    D: Dimension + Copy,
    <D as Dimension>::Smaller: Copy,
    S: ndarray::RawDataClone<Elem = A> + Copy,
    A: Copy,
{
}

impl<A, S, D> PartialEq for ParamsBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> Eq for ParamsBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: Eq,
{
}

impl<A, S, D> core::fmt::Debug for ParamsBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("bias", self.bias())
            .field("weights", self.weights())
            .finish()
    }
}

// impl<A, S, D> PartialOrd for ModelParams<S, D>
// where
//     D: Dimension,
//     S: Data<Elem = A>,
//     A: PartialOrd,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
//         match self.bias.iter().partial_cmp(&other.bias.iter()) {
//             Some(core::cmp::Ordering::Equal) => self.weights.iter().partial_cmp(&other.weights.iter()),
//             other => other,
//         }
//     }
// }
