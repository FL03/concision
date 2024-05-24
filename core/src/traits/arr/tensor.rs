/*
    Appellation: generator <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Dimensional;
use nd::iter::{Iter, IterMut};
use nd::prelude::*;
use nd::{Data, DataMut, DataOwned, OwnedRepr, RawData};
use num::{One, Zero};

pub trait NdArray<A, D>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    fn as_slice(&self) -> &[A];

    fn as_mut_slice(&mut self) -> &mut [A];

    fn iter(&self) -> Iter<'_, A, D>;

    fn iter_mut(&mut self) -> IterMut<'_, A, D>;

    fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(&A) -> A;

    fn mapv<F>(&mut self, f: F)
    where
        A: Clone,
        F: FnMut(A) -> A;
}

pub trait NdIter<A, D>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    fn iter(&self) -> Iter<'_, A, D>;

    fn iter_mut(&mut self) -> IterMut<'_, A, D>;
}

/// [NdBuilder] describes common creation routines for [ArrayBase](ndarray::ArrayBase)
pub trait NdBuilder<A = f64, D = Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    /// Create a new array with the given shape whose elements are set to the default value of the element type.
    fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn fill<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
}

pub trait NdBuilderExt<A = f64, D = Ix2>: NdBuilder<A, D>
where
    D: Dimension,
    Self: Dimensional<D, Pattern = D::Pattern> + Sized,
{
    fn default_like<Sh>(&self) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::default(self.dim())
    }

    fn fill_like<Sh>(&self, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::fill(self.dim(), elem)
    }

    fn ones_like<Sh>(&self) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::ones(self.dim())
    }

    fn zeros_like<Sh>(&self) -> Self
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

pub trait View<A = f64, D = Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;
    type Output;

    fn view(&self) -> Self::Output
    where
        A: Clone,
        Self::Data: Data;
}
pub trait ViewMut<A = f64, D = Ix2>: View<A, D>
where
    D: Dimension,
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, A, D>
    where
        A: Clone,
        Self::Data: DataMut;
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
