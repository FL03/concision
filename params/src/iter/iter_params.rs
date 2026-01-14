/*
    Appellation: iter <module>
    Contrib: @FL03
*/
//! Iterators for parameters within a neural network
use ndarray::Dimension;
use ndarray::iter::{AxisIter, AxisIterMut};
use ndarray::iter::{Iter as NdIter, IterMut as NdIterMut};

pub(crate) type ItemRef<'a, A, D> = (
    <AxisIter<'a, A, <D as Dimension>::Smaller> as Iterator>::Item,
    &'a A,
);
pub(crate) type ItemMut<'a, A, D> = (
    <AxisIterMut<'a, A, <D as Dimension>::Smaller> as Iterator>::Item,
    &'a mut A,
);
/// The [`ParamsIter`] defines a iterator over owned parameters within a neural network by
/// zipping together an axis iterator over the columns of the weights and an iterator over the
/// bias.
pub struct ParamsIter<'a, A, D>
where
    D: Dimension,
{
    pub(crate) weights: AxisIter<'a, A, D::Smaller>,
    pub(crate) bias: NdIter<'a, A, D::Smaller>,
}
/// The [`ParamsIterMut`] type provides a mutable iterator over the parameters of a neural
/// network layer by zipping together a mutable axis iterator over the columns of the weights
/// and a mutable iterator over the bias.
pub struct ParamsIterMut<'a, A, D>
where
    D: Dimension,
{
    pub(crate) weights: AxisIterMut<'a, A, D::Smaller>,
    pub(crate) bias: NdIterMut<'a, A, D::Smaller>,
}

/*
 ************* Implementations *************
*/
impl<'a, A, D> Iterator for ParamsIter<'a, A, D>
where
    D: Dimension,
{
    type Item = ItemRef<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.weights.next(), self.bias.next()) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        }
    }
}

impl<'a, A, D> ExactSizeIterator for ParamsIter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.weights.len()
    }
}

impl<'a, A, D> Iterator for ParamsIterMut<'a, A, D>
where
    D: Dimension,
{
    type Item = ItemMut<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.weights.next(), self.bias.next()) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        }
    }
}
