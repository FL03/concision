/*
    Appellation: tensor <module>
    Contrib: @FL03
*/
use crate::traits::Scalar;
use ndarray::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use num_traits::{One, Zero};

pub trait Tensor<S, D>
where
    S: RawData<Elem = Self::Scalar>,
    D: Dimension,
{
    type Scalar;
    type Container<U: RawData, V: Dimension>;
    /// Create a new tensor with the given shape and a function to fill it
    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Container<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> Self::Scalar,
        Self: Sized;
    /// Create a new tensor with the given shape and value
    fn from_shape_with_value<Sh>(shape: Sh, value: Self::Scalar) -> Self::Container<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized;
    /// Create a new tensor with the given shape and all values set to their default
    fn default<Sh>(shape: Sh) -> Self::Container<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        Self::Scalar: Default,
    {
        Self::from_shape_with_value(shape, Self::Scalar::default())
    }
    /// create a new tensor with the given shape and all values set to one
    fn ones<Sh>(shape: Sh) -> Self::Container<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        Self::Scalar: Clone + One,
    {
        Self::from_shape_with_value(shape, Self::Scalar::one())
    }
    /// create a new tensor with the given shape and all values set to zero
    fn zeros<Sh>(shape: Sh) -> Self::Container<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        Self::Scalar: Clone + Zero,
    {
        Self::from_shape_with_value(shape, <Self as Tensor<S, D>>::Scalar::zero())
    }
    /// returns a reference to the data of the object
    fn data(&self) -> &Self::Container<S, D>;
    /// returns a mutable reference to the data of the object
    fn data_mut(&mut self) -> &mut Self::Container<S, D>;
    /// returns the number of dimensions of the object
    fn dim(&self) -> D::Pattern;
    /// returns the shape of the object
    fn raw_dim(&self) -> D;
    /// returns the shape of the object
    fn shape(&self) -> &[usize];
    /// sets the data of the object
    fn set_data(&mut self, data: Self::Container<S, D>) -> &mut Self {
        *self.data_mut() = data;
        self
    }
}

impl<A, S, D> Tensor<S, D> for ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    A: Scalar,
    D: Dimension,
{
    type Scalar = A;
    type Container<U: RawData, V: Dimension> = ArrayBase<U, V>;

    fn from_shape_with_value<Sh>(shape: Sh, value: Self::Scalar) -> Self::Container<S, D>
    where
        Self: Sized,
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        Self::Container::<S, D>::from_elem(shape, value)
    }

    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Container<S, D>
    where
        Self: Sized,
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> Self::Scalar,
    {
        Self::Container::<S, D>::from_shape_fn(shape, f)
    }

    fn data(&self) -> &Self::Container<S, D> {
        self
    }

    fn data_mut(&mut self) -> &mut Self::Container<S, D> {
        self
    }

    fn dim(&self) -> D::Pattern {
        self.dim()
    }

    fn raw_dim(&self) -> D {
        self.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
