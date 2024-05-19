/*
   Appellation: convert <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::Axis;
use nd::{ArrayBase, Dimension, RawData};

pub trait Dimensional<D> {
    type Pattern;

    fn dim(&self) -> Self::Pattern;

    fn raw_dim(&self) -> D;

    fn shape(&self) -> &[usize];
}

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

pub trait IsSquare {
    fn is_square(&self) -> bool;
}

/*
 ******** implementations ********
*/
impl<S, D> Dimensional<D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData,
{
    type Pattern = D::Pattern;

    fn shape(&self) -> &[usize] {
        ArrayBase::shape(self)
    }

    fn dim(&self) -> Self::Pattern {
        ArrayBase::dim(self)
    }

    fn raw_dim(&self) -> D {
        ArrayBase::raw_dim(self)
    }
}

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}

impl<S, D> IsSquare for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData,
{
    fn is_square(&self) -> bool {
        let first = self.shape().first().unwrap();
        self.shape().iter().all(|x| x == first)
    }
}
