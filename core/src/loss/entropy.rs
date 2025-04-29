/*
    Appellation: loss <module>
    Contrib: @FL03
*/

/// A trait for computing the cross-entropy loss of a tensor or array
pub trait CrossEntropy {
    type Output;

    fn cross_entropy(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/

use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> CrossEntropy for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn cross_entropy(&self) -> Self::Output {
        self.mapv(|x| -x.ln()).mean().unwrap()
    }
}
