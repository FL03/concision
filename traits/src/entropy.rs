/*
    Appellation: entropy <module>
    Created At: 2025.11.26:13:13:33
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
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> CrossEntropy for ArrayBase<S, D, A>
where
    A: 'static + Float + FromPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn cross_entropy(&self) -> Self::Output {
        self.mapv(|x| -x.ln()).mean().unwrap()
    }
}
