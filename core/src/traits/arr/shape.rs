/*
    Appellation: shape <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, Dimension, RawData};

pub trait Dimensional<D> {
    type Pattern;

    fn dim(&self) -> Self::Pattern;

    fn raw_dim(&self) -> D;

    fn shape(&self) -> &[usize];
}

/*
 ********* Implementations *********
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
