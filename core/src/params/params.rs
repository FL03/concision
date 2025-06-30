/*
    Appellation: params <module>
    Contrib: @FL03
*/
use ndarray::{
    ArrayBase, Axis, Data, DataMut, DataOwned, Dimension, RawData, RemoveAxis, ShapeArg,
    ShapeBuilder,
};

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
    /// returns a new instance of the [`ParamsBase`] using the initialization routine
    pub fn init_from_fn<Sh, F>(shape: Sh, init: F) -> Self
    where
        A: Clone,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        F: Fn() -> A,
    {
        let shape = shape.into_shape_with_order();
        let bshape = shape.raw_dim().remove_axis(Axis(0));
        // initialize the bias and weights using the provided function for each element
        let bias = ArrayBase::from_shape_fn(bshape, |_| init());
        let weights = ArrayBase::from_shape_fn(shape, |_| init());
        // create a new instance from the generated bias and weights
        Self::new(bias, weights)
    }
    /// returns a new instance of the [`ParamsBase`] initialized use the given shape_function
    pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> Self
    where
        A: Clone,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        D::Smaller: Dimension + ShapeArg,
        F: Fn(<D::Smaller as Dimension>::Pattern) -> A + Fn(<D as Dimension>::Pattern) -> A,
    {
        let shape = shape.into_shape_with_order();
        let bdim = shape.raw_dim().remove_axis(Axis(0));
        let bias = ArrayBase::from_shape_fn(bdim, |s| f(s));
        let weights = ArrayBase::from_shape_fn(shape, |s| f(s));
        Self::new(bias, weights)
    }
    /// create a new instance of the [`ParamsBase`] with the given bias used the default weights
    pub fn from_bias<Sh>(shape: Sh, bias: ArrayBase<S, D::Smaller>) -> Self
    where
        A: Clone + Default,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let weights = ArrayBase::from_elem(shape, A::default());
        Self::new(bias, weights)
    }
    /// create a new instance of the [`ParamsBase`] with the given weights used the default
    /// bias
    pub fn from_weights<Sh>(shape: Sh, weights: ArrayBase<S, D>) -> Self
    where
        A: Clone + Default,
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape_with_order();
        let dim_bias = shape.raw_dim().remove_axis(Axis(0));
        let bias = ArrayBase::from_elem(dim_bias, A::default());
        Self::new(bias, weights)
    }
    /// create a new instance of the [`ParamsBase`] from the given shape and element;
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
    #[allow(clippy::should_implement_trait)]
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
    /// returns a new [`ParamsBase`] instance with the same paramaters, but using a shared
    /// representation of the data;
    pub fn to_shared(&self) -> ParamsBase<ndarray::OwnedArcRepr<A>, D>
    where
        A: Clone,
        S: Data,
    {
        ParamsBase::new(self.bias().to_shared(), self.weights().to_shared())
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
