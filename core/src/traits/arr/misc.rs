/*
   Appellation: convert <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::Axis;

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

pub trait NdArray<T> {
    type Dim;

    fn as_slice(&self) -> &[T];

    fn dim(&self) -> Self::Dim;

    fn shape(&self) -> &[usize];

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn is_scalar(&self) -> bool {
        self.shape().is_empty()
    }
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

use nd::{ArrayBase, Dimension};

impl<A, S, D> NdArray<A> for ArrayBase<S, D>
where
    S: nd::Data<Elem = A>,
    D: Dimension,
{
    type Dim = D;

    fn as_slice(&self) -> &[A] {
        ArrayBase::as_slice(self).unwrap()
    }

    fn dim(&self) -> D {
        self.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
