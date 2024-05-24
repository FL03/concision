/*
    Appellation: ndview <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
/*
    Appellation: ndarray <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::prelude::*;
use nd::{Data, DataMut, DataOwned, OwnedRepr, RawData};

pub trait AsOwned<A, S, D = Ix2>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output;

    fn into_owned(self) -> Self::Output
    where
        A: Clone,
        S: Data;

    fn to_owned(&self) -> Self::Output
    where
        A: Clone,
        S: Data;
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

pub trait NdView<A = f64, S = OwnedRepr<A>, D = Ix2>: AsOwned<A, S, D> + AsShared<S, D>
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
impl<A, S, D> AsOwned<A, S, D> for ArrayBase<S, D>
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
