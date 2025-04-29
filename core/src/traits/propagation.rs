/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// [Backward] propagate a delta through the system;
pub trait Backward<X, Delta = X> {
    type Elem;
    type Output;

    fn backward(
        &mut self,
        input: &X,
        delta: &Delta,
        gamma: Self::Elem,
    ) -> crate::Result<Self::Output>;
}

/// This trait denotes entities capable of performing a single forward step
pub trait Forward<Rhs> {
    type Output;
    /// a single forward step
    fn forward(&self, input: &Rhs) -> crate::Result<Self::Output>;
    /// this method enables the forward pass to be generically _activated_ using some closure.
    /// This is useful for isolating the logic of the forward pass from that of the activation
    /// function and is often used by layers and models.
    fn forward_then<F>(&self, input: &Rhs, then: F) -> crate::Result<Self::Output>
    where
        F: FnOnce(Self::Output) -> Self::Output,
    {
        self.forward(input).map(then)
    }
}

/*
 ************* Implementations *************
*/

use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Data, Dimension};
// impl<X, Y, Dx, A, S, D> Backward<X, Y> for ArrayBase<S, D>
// where
//     A: LinalgScalar + FromPrimitive,
//     D: Dimension,
//     S: DataMut<Elem = A>,
//     Dx: core::ops::Mul<A, Output = Dx>,
//     for<'a> X: Dot<Y, Output = Dx>,
//     for<'a> &'a Self: core::ops::Add<Dx, Output = Self>,

// {
//     type Elem = A;
//     type Output = ();

//     fn backward(
//         &mut self,
//         input: &X,
//         delta: &Y,
//         gamma: Self::Elem,
//     ) -> crate::Result<Self::Output> {
//         let grad = input.dot(delta);
//         let next = &self + grad * gamma;
//         self.assign(&next)?;
//         Ok(())

//     }
// }

impl<X, Y, A, S, D> Forward<X> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> X: Dot<ArrayBase<S, D>, Output = Y>,
{
    type Output = Y;

    fn forward(&self, input: &X) -> crate::Result<Self::Output> {
        let output = input.dot(self);
        Ok(output)
    }
}
