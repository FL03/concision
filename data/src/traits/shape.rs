/*
    Appellation: shape <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, Dimension, RawData};

pub trait IntoPattern {
    type Pattern;

    fn into_pattern(self) -> Self::Pattern;
}

/// [Dimensional] provides a common interface for containers to access their shape and dimension.
pub trait Dimensional {
    const RANK: Option<usize> = None;

    type Dim: IntoPattern;

    fn dim(&self) -> <Self::Dim as IntoPattern>::Pattern {
        self.raw_dim().into_pattern()
    }

    fn is_scalar(&self) -> bool {
        self.rank() == 0 || self.shape().iter().all(|x| *x == 1)
    }

    fn rank(&self) -> usize {
        Self::RANK.unwrap_or(self.shape().len())
    }

    fn raw_dim(&self) -> Self::Dim;

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn shape(&self) -> &[usize];
}

/*
 ******** implementations ********
*/
impl<D> IntoPattern for D
where
    D: Dimension,
{
    type Pattern = D::Pattern;

    fn into_pattern(self) -> Self::Pattern {
        Dimension::into_pattern(self)
    }
}

// impl<D> Dimensional for D
// where
//     D: Dimension + IntoPattern,
// {
//     type Dim = D;

//     fn dim(&self) -> D::Pattern {
//         self.clone().into_pattern()
//     }

//     fn raw_dim(&self) -> D {
//         self.clone()
//     }

//     fn shape(&self) -> &[usize] {
//         D::slice(self)
//     }
// }

impl<S, D> Dimensional for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData,
{
    const RANK: Option<usize> = D::NDIM;
    type Dim = D;

    fn dim(&self) -> D::Pattern {
        ArrayBase::dim(self)
    }

    fn raw_dim(&self) -> D {
        ArrayBase::raw_dim(self)
    }

    fn shape(&self) -> &[usize] {
        ArrayBase::shape(self)
    }

    fn size(&self) -> usize {
        ArrayBase::len(self)
    }
}
