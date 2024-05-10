/*
   Appellation: convert <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::Axis;
use nd::{ArrayBase, Dimension, RawData};

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

pub trait IsSquare {
    fn is_square(&self) -> bool;
}

/*
 ******** implementations ********
*/
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
        self.shape().iter().all(|&x| x == self.shape()[0])
    }
}
