/*
    appellation: tensor <module>
    authors: @FL03
*/
use ndarray::{ArrayBase, Data, DataMut, DataOwned, Dimension, NdIndex, RawData, ShapeBuilder};
use num_traits::{One, Zero};

/// the [`TensorBase`] struct is the base type for all tensors in the library.
#[repr(transparent)]
pub struct TensorBase<S, D>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) store: ArrayBase<S, D>,
}

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new [`TensorBase`] from the given store.
    pub const fn from_ndarray(store: ArrayBase<S, D>) -> Self {
        Self { store }
    }
    /// create a new [`TensorBase`] from the given shape and a function to fill it.
    pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
    {
        Self {
            store: ArrayBase::from_shape_fn(shape, f),
        }
    }
    /// create a new [`TensorBase`] from the given shape and a function to fill it.
    pub fn from_fn_with_shape<Sh, F>(shape: Sh, f: F) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        F: Fn() -> A,
    {
        Self::from_shape_fn(shape, |_| f())
    }
    /// returns a new instance of the [`TensorBase`] with the given shape and values initialized
    /// to zero.
    pub fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_fn_with_shape(shape, A::one)
    }
    /// returns a new instance of the [`TensorBase`] with the given shape and values initialized
    /// to zero.
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_fn_with_shape(shape, A::zero)
    }
    /// returns a reference to the element at the given index, if any
    pub fn get<Ix>(&self, index: Ix) -> Option<&A>
    where
        S: Data,
        Ix: NdIndex<D>,
    {
        self.store().get(index)
    }
    /// returns a mutable reference to the element at the given index, if any
    pub fn get_mut<Ix>(&mut self, index: Ix) -> Option<&mut A>
    where
        S: DataMut,
        Ix: NdIndex<D>,
    {
        self.store_mut().get_mut(index)
    }
    /// applies the function to every element within the tensor
    pub fn map<F, B>(&self, f: F) -> super::Tensor<B, D>
    where
        A: Clone,
        S: Data,
        F: FnMut(A) -> B,
    {
        TensorBase {
            store: self.store().mapv(f),
        }
    }
    /// this method applies the function to the store, capturing the result in a new tensor.
    pub(crate) fn mapd<F, U, S2, D2>(&self, f: F) -> TensorBase<S2, D2>
    where
        D2: Dimension,
        S2: RawData<Elem = U>,
        F: FnOnce(&ArrayBase<S, D>) -> ArrayBase<S2, D2>,
    {
        TensorBase {
            store: f(self.store()),
        }
    }
}

#[doc(hidden)]
#[allow(dead_code)]
impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns an immutable reference to the store of the tensor
    pub(crate) const fn store(&self) -> &ArrayBase<S, D> {
        &self.store
    }
    /// returns a mutable reference to the store of the tensor
    pub(crate) const fn store_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.store
    }
    /// update the current store and return a mutable reference to self
    pub(crate) fn set_store(&mut self, store: ArrayBase<S, D>) -> &mut Self {
        self.store = store;
        self
    }
}
