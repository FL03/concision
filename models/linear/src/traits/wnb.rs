/*
    Appellation: wnb <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

pub trait WnB<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    fn bias(&self) -> Option<&ArrayBase<S, D::Smaller>>;

    fn bias_mut(&mut self) -> Option<&mut ArrayBase<S, D::Smaller>>;

    fn weight(&self) -> &ArrayBase<S, D>;

    fn weight_mut(&mut self) -> &mut ArrayBase<S, D>;
}

pub trait Dimensional<D> {
    type Pattern;

    fn dim(&self) -> Self::Pattern;

    fn raw_dim(&self) -> D;

    fn shape(&self) -> &[usize];
}

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

pub trait IsBiased {
    fn is_biased(&self) -> bool;
}

/*
 ********* Implementations *********
*/
