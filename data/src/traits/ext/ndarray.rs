/*
    Appellation: ndarray <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::iter::{Iter, IterMut};
use nd::{Dimension, RawData};

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

/*
 ************* Implementations *************
*/
