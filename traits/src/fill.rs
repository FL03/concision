/*
   Appellation: convert <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::*;
use ndarray::{DataMut, RawData};

/// This trait is used to fill an array with a value based on a mask.
/// The mask is a boolean array of the same shape as the array.
pub trait MaskFill<A, D>
where
    D: Dimension,
{
    type Output;

    fn masked_fill(&self, mask: &Array<bool, D>, value: A) -> Self::Output;
}

/// [`IsSquare`] is a trait for checking if the layout, or dimensionality, of a tensor is
/// square.
pub trait IsSquare {
    fn is_square(&self) -> bool;
}

/*
 ******** implementations ********
*/

impl<A, S, D> MaskFill<A, D> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: DataMut<Elem = A>,
    Self: Clone,
{
    type Output = ArrayBase<S, D, A>;

    fn masked_fill(&self, mask: &Array<bool, D>, value: A) -> Self::Output {
        let mut arr = self.clone();
        arr.zip_mut_with(mask, |x, &m| {
            if m {
                *x = value.clone();
            }
        });
        arr
    }
}

impl<A, S, D> IsSquare for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn is_square(&self) -> bool {
        let first = self.shape().first().unwrap();
        self.shape().iter().all(|x| x == first)
    }
}
