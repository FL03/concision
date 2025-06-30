/*
    appellation: activate <module>
    authors: @FL03
*/
use super::unary::*;

use crate::Apply;
use num_complex::ComplexFloat;
use num_traits::One;

/// The [`CncActivate`] trait builds off of the [`Apply`] trait, extending it to provide 
/// various activation functions and their derivatives.
pub trait CncActivate<U>: Apply<U> {
    /// the linear activation function is essentially a passthrough function, simply cloning 
    /// the content.
    fn linear(&self) -> Self::Cont<U>
    where
        U: Clone,
    {
        self.apply(|x| x.clone())
    }

    fn linear_derivative(&self) -> Self::Cont<U::Output>
    where
        U: One,
    {
        self.apply(|_| <U>::one())
    }

    fn heavyside(&self) -> Self::Cont<U::Output>
    where
        U: Heavyside,
    {
        self.apply(|x| x.heavyside())
    }

    fn heavyside_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Heavyside,
    {
        self.apply(|x| x.heavyside_derivative())
    }

    fn relu(&self) -> Self::Cont<U::Output>
    where
        U: ReLU,
    {
        self.apply(|x| x.relu())
    }

    fn relu_derivative(&self) -> Self::Cont<U::Output>
    where
        U: ReLU,
    {
        self.apply(|x| x.relu_derivative())
    }

    fn sigmoid(&self) -> Self::Cont<U::Output>
    where
        U: Sigmoid,
    {
        self.apply(|x| x.sigmoid())
    }

    fn sigmoid_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Sigmoid,
    {
        self.apply(|x| x.sigmoid_derivative())
    }

    fn tanh(&self) -> Self::Cont<U::Output>
    where
        U: Tanh,
    {
        self.apply(|x| x.tanh())
    }

    fn tanh_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Tanh,
    {
        self.apply(|x| x.tanh_derivative())
    }

    fn sigmoid_complex(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.apply(|x| U::one() / (U::one() + (-x).exp()))
    }

    fn sigmoid_complex_derivative(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.apply(|x| {
            let s = U::one() / (U::one() + (-x).exp());
            s * (U::one() - s)
        })
    }

    fn tanh_complex(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.apply(|x| x.tanh())
    }

    fn tanh_complex_derivative(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.apply(|x| {
            let s = x.tanh();
            U::one() - s * s
        })
    }
}

/*
 ************* Implementations *************
*/
impl<U, S> CncActivate<U> for S where S: Apply<U> {}
