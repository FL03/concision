/*
    Appellation: tensor <module>
    Contrib: @FL03
*/
use crate::traits::Scalar;
use ndarray::{
    ArrayBase, Axis, DataMut, DataOwned, Dimension, OwnedRepr, RawData, RemoveAxis, ShapeBuilder,
};
use num::Signed;
use num_traits::{FromPrimitive, One, Zero, Pow};

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
    /// returns a new tensor with the same shape as the object and the given function applied
    /// to each element
    fn apply<F, B>(&self, f: F) -> Self::Container<OwnedRepr<B>, D>
    where
        F: FnMut(Self::Scalar) -> B;
    /// returns a new tensor with the same shape as the object and the given function applied
    fn apply_mut<F>(&mut self, f: F)
    where
        S: DataMut,
        F: FnMut(Self::Scalar) -> Self::Scalar;

    fn axis_iter(&self, axis: usize) -> ndarray::iter::AxisIter<'_, Self::Scalar, D::Smaller>
    where
        D: RemoveAxis;

    fn iter(&self) -> ndarray::iter::Iter<'_, Self::Scalar, D>;

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, Self::Scalar, D>
    where
        S: DataMut;

    fn mean(&self) -> Self::Scalar
    where
        Self::Scalar: Scalar,
    {
        let sum = self.sum();
        let count = self.iter().count();
        sum / Self::Scalar::from_usize(count).unwrap()
    }

    fn sum(&self) -> Self::Scalar
    where
        Self::Scalar: Clone + core::iter::Sum,
    {
        self.iter().cloned().sum()
    }

    fn pow2(&self) -> Self::Container<OwnedRepr<Self::Scalar>, D>
    where
        Self::Scalar: Scalar,
    {
        let two = Self::Scalar::from_usize(2).unwrap();
        self.apply(|x| x.pow(two))
    }

    fn abs(&self) -> Self::Container<OwnedRepr<Self::Scalar>, D>
    where
        Self::Scalar: Signed,
    {
        self.apply(|x| x.abs())
    }

    fn neg(&self) -> Self::Container<OwnedRepr<Self::Scalar>, D>
    where
        Self::Scalar: core::ops::Neg<Output = Self::Scalar>,
    {
        self.apply(|x| -x)
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

    fn apply<F, B>(&self, f: F) -> Self::Container<OwnedRepr<B>, D>
    where
        F: FnMut(Self::Scalar) -> B,
    {
        self.mapv(f)
    }

    fn apply_mut<F>(&mut self, f: F)
    where
        F: FnMut(Self::Scalar) -> Self::Scalar,
        S: DataMut,
    {
        self.mapv_inplace(f)
    }

    fn iter(&self) -> ndarray::iter::Iter<'_, Self::Scalar, D> {
        self.iter()
    }
    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, Self::Scalar, D>
    where
        S: DataMut,
    {
        self.iter_mut()
    }
    fn axis_iter(&self, axis: usize) -> ndarray::iter::AxisIter<'_, Self::Scalar, D::Smaller>
    where
        D: RemoveAxis,
    {
        self.axis_iter(Axis(axis))
    }
}
