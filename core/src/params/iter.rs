/*
    Appellation: iter <module>
    Contrib: @FL03
*/
use ndarray::Dimension;
use ndarray::iter::AxisIter;
use ndarray::iter::Iter as NdIter;

pub type ItemMut<'a, A, D> = (
    <AxisIter<'a, A, <D as Dimension>::Smaller> as Iterator>::Item,
    &'a mut A,
);

pub struct Iter<'a, A, D>
where
    D: Dimension,
{
    pub(crate) weights: AxisIter<'a, A, D::Smaller>,
    pub(crate) bias: NdIter<'a, A, D::Smaller>,
}

impl<'a, A, D> Iter<'a, A, D>
where
    D: Dimension,
{
    pub fn new(weights: AxisIter<'a, A, D::Smaller>, bias: NdIter<'a, A, D::Smaller>) -> Self {
        Self { weights, bias }
    }
}

impl<'a, A, D> Iterator for Iter<'a, A, D>
where
    D: Dimension,
{
    type Item = (
        <AxisIter<'a, A, <D as Dimension>::Smaller> as Iterator>::Item,
        &'a A,
    );

    fn next(&mut self) -> Option<Self::Item> {
        match (self.weights.next(), self.bias.next()) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        }
    }
}

impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.weights.len()
    }
}
