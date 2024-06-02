/*
    Appellation: activate <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::utils::*;
pub use self::{binary::*, linear::*, nonlinear::*};

pub(crate) mod utils;

pub mod binary;
pub mod linear;
pub mod nonlinear;

pub(crate) mod prelude {
    pub use super::binary::*;
    pub use super::linear::*;
    pub use super::nonlinear::*;
    pub use super::utils::*;
    pub use super::{Activate, NdActivate};
}

use nd::prelude::*;
use nd::{Data, DataMut, RemoveAxis, ScalarOperand};
use num::complex::ComplexFloat;
use num::traits::{One, Zero};

/// [Activate] designates a function or structure that can be used
/// as an activation function for a neural network.
///
/// The trait enables implemented models to employ various activation
/// functions either as a pure function or as a structure.
pub trait Activate<T> {
    type Output;

    fn activate(&self, args: T) -> Self::Output;
}

pub trait NdActivate<A, D>
where
    A: ScalarOperand,
    D: Dimension,
{
    type Data: Data<Elem = A>;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: FnMut(A) -> B;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
        Self::Data: DataMut<Elem = A>;

    fn linear(&self) -> Array<A, D>
    where
        A: Clone,
    {
        self.activate(|x| x.clone())
    }

    fn heavyside(&self) -> Array<A, D>
    where
        A: One + PartialOrd + Zero,
    {
        self.activate(heavyside)
    }

    fn relu(&self) -> Array<A, D>
    where
        A: PartialOrd + Zero,
    {
        self.activate(relu)
    }

    fn sigmoid(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(sigmoid)
    }

    fn softmax(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        let exp = self.activate(ComplexFloat::exp);
        &exp / exp.sum()
    }

    fn softmax_axis(&self, axis: usize) -> Array<A, D>
    where
        A: ComplexFloat,
        D: RemoveAxis,
    {
        let exp = self.activate(ComplexFloat::exp);
        let axis = Axis(axis);
        &exp / &exp.sum_axis(axis)
    }

    fn tanh(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(tanh)
    }
}
/*
 ************* Implementations *************
*/

activator!(LinearActor::<T>(T::clone) where T: Clone);

impl<F, U, V> Activate<U> for F
where
    F: Fn(U) -> V,
{
    type Output = V;

    fn activate(&self, args: U) -> Self::Output {
        self(args)
    }
}

impl<U, V> Activate<U> for Box<dyn Activate<U, Output = V>> {
    type Output = V;

    fn activate(&self, args: U) -> Self::Output {
        self.as_ref().activate(args)
    }
}

impl<A, S, D> NdActivate<A, D> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Data = S;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: FnMut(A) -> B,
    {
        self.mapv(f)
    }

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        S: DataMut,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
