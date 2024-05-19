/*
    Appellation: generator <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Dimensional;
use nd::prelude::*;
use nd::{Data, DataMut, DataOwned, OwnedRepr, RawData};
use num::{One, Zero};

/// This trait describes the basic operations for any n-dimensional container.
pub trait NdContainer<A = f64, D = Ix2>: Dimensional<D> {
    type Data;

    fn as_slice(&self) -> &[A];

    fn as_mut_slice(&mut self) -> &mut [A];
}

/// [NdBuilder] describes common creation routines for [ArrayBase](ndarray::ArrayBase)
pub trait NdBuilder<A = f64, D = Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;
    type Store;

    /// Create a new array with the given shape whose elements are set to the default value of the element type.
    fn default<Sh>(shape: Sh) -> Self::Store
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn fill<Sh>(shape: Sh, elem: A) -> Self::Store
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn ones<Sh>(shape: Sh) -> Self::Store
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn zeros<Sh>(shape: Sh) -> Self::Store
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
}

pub trait NdBuilderExt<A = f64, D = Ix2>: Dimensional<D, Pattern = D::Pattern> + NdBuilder<A, D>
where
    D: Dimension,
{
    fn default_like<Sh>(&self) -> Self::Store
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::default(self.dim())
    }

    fn fill_like<Sh>(&self, elem: A) -> Self::Store
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::fill(self.dim(), elem)
    }

    fn ones_like<Sh>(&self) -> Self::Store
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::ones(self.dim())
    }

    fn zeros_like<Sh>(&self) -> Self::Store
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::zeros(self.dim())
    }
}

pub trait AsOwned<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    type Output;

    fn into_owned(self) -> Self::Output
    where
        S: Data,
        S::Elem: Clone;

    fn to_owned(&self) -> Self::Output
    where
        S: Data,
        S::Elem: Clone;
}

pub trait AsShared<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    type Output;

    fn into_shared(self) -> Self::Output
    where
        S: DataOwned,
        S::Elem: Clone;

    fn to_shared(&self) -> Self::Output
    where
        S: DataOwned,
        S::Elem: Clone;
}

pub trait NdView<A = f64, S = OwnedRepr<A>, D = Ix2>: AsOwned<S, D> + AsShared<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn view(&self) -> ArrayView<'_, A, D>
    where
        A: Clone,
        S: Data;

    fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
    where
        A: Clone,
        S: DataMut;
}

/*
 ************* Implementations *************
*/
impl<A, S, D> NdBuilder<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Data = S;
    type Store = ArrayBase<S, D>;

    fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::default(shape)
    }

    fn fill<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        ArrayBase::from_elem(shape, elem)
    }

    fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::ones(shape)
    }

    fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::zeros(shape)
    }
}

impl<U, A, D> NdBuilderExt<A, D> for U
where
    U: Dimensional<D, Pattern = D::Pattern> + NdBuilder<A, D>,
    D: Dimension,
{
}

impl<A, S, D> AsOwned<S, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = Array<A, D>;

    fn into_owned(self) -> Self::Output
    where
        A: Clone,
        S: Data,
    {
        self.into_owned()
    }

    fn to_owned(&self) -> Self::Output
    where
        A: Clone,
        S: Data,
    {
        self.to_owned()
    }
}

impl<A, S, D> AsShared<S, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = ArcArray<A, D>;

    fn into_shared(self) -> Self::Output
    where
        A: Clone,
        S: DataOwned,
    {
        self.into_shared()
    }

    fn to_shared(&self) -> Self::Output
    where
        A: Clone,
        S: DataOwned,
    {
        self.to_shared()
    }
}

impl<A, S, D> NdView<A, S, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn view(&self) -> ArrayView<'_, A, D>
    where
        A: Clone,
        S: Data,
    {
        self.view()
    }

    fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
    where
        A: Clone,
        S: DataMut,
    {
        self.view_mut()
    }
}
