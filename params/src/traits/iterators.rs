/*
    Appellation: iterators <module>
    Created At: 2025.12.14:11:02:25
    Contrib: @FL03
*/
use ndarray::iter as nditer;
use ndarray::{ArrayBase, Data, DataMut, Dimension};

pub trait NdIter<A, D>
where
    D: Dimension,
{
    type Iter<'b, T>
    where
        T: 'b,
        Self: 'b;
    /// returns an iterator over the weights;
    fn nditer(&self) -> Self::Iter<'_, A>;
}

pub trait NdIterMut<A, D>
where
    D: Dimension,
{
    type IterMut<'b, T>
    where
        T: 'b,
        Self: 'b;
    /// returns a mutable iterator over the weights
    fn nditer_mut(&mut self) -> Self::IterMut<'_, A>;
}

/*
 ************* Implementations *************
*/

impl<A, S, D> NdIter<A, D> for ArrayBase<S, D, A>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    type Iter<'b, T>
        = nditer::Iter<'b, T, D>
    where
        T: 'b,
        Self: 'b;

    fn nditer(&self) -> Self::Iter<'_, A> {
        self.iter()
    }
}

impl<A, S, D> NdIterMut<A, D> for ArrayBase<S, D, A>
where
    S: DataMut<Elem = A>,
    D: Dimension,
{
    type IterMut<'b, T>
        = nditer::IterMut<'b, T, D>
    where
        T: 'b,
        Self: 'b;

    fn nditer_mut(&mut self) -> Self::IterMut<'_, A> {
        self.iter_mut()
    }
}
