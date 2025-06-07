/*
    Appellation: traits <activate>
    Contrib: @FL03
*/

use ndarray::prelude::*;
use ndarray::{Data, DataMut, ScalarOperand};
use num::complex::ComplexFloat;

macro_rules! unary {
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };

    (@impl $name:ident::$call:ident(self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(self) -> Self::Output;

                fn [<$call _derivative>](self) -> Self::Output;
            }
        }

    };
    (@impl $name:ident::$call:ident(&self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(&self) -> Self::Output;

                fn [<$call _derivative>](&self) -> Self::Output;
            }
        }
    };
}

unary! {
    Heavyside::heavyside(self),
    LinearActivation::linear(self),
    Sigmoid::sigmoid(self),
    Softmax::softmax(&self),
    ReLU::relu(&self),
    Tanh::tanh(&self),
}

pub trait SoftmaxAxis: Softmax {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

/// A trait defining the manner in which a particular entity can be activated.
pub trait Activate<A> {
    type Cont<B>;

    fn activate<V, F>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(A) -> V;
}
/// A trait for establishing a common mechanism to activate entities in-place.
pub trait ActivateMut<A> {
    type Cont<B>;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A;
}
/// This trait extends the [`Activate`] trait with a number of additional activation functions
/// and their derivatives. _**Note:**_ this trait is automatically implemented for any type
/// that implements the [`Activate`] trait eliminating the need to implement it manually.
pub trait ActivateExt<U>: Activate<U> {
    fn linear(&self) -> Self::Cont<U::Output>
    where
        U: LinearActivation,
    {
        self.activate(|x| x.linear())
    }

    fn linear_derivative(&self) -> Self::Cont<U::Output>
    where
        U: LinearActivation,
    {
        self.activate(|x| x.linear_derivative())
    }

    fn heavyside(&self) -> Self::Cont<U::Output>
    where
        U: Heavyside,
    {
        self.activate(|x| x.heavyside())
    }

    fn heavyside_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Heavyside,
    {
        self.activate(|x| x.heavyside_derivative())
    }

    fn relu(&self) -> Self::Cont<U::Output>
    where
        U: ReLU,
    {
        self.activate(|x| x.relu())
    }

    fn relu_derivative(&self) -> Self::Cont<U::Output>
    where
        U: ReLU,
    {
        self.activate(|x| x.relu_derivative())
    }

    fn sigmoid(&self) -> Self::Cont<U::Output>
    where
        U: Sigmoid,
    {
        self.activate(|x| x.sigmoid())
    }

    fn sigmoid_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Sigmoid,
    {
        self.activate(|x| x.sigmoid_derivative())
    }

    fn tanh(&self) -> Self::Cont<U::Output>
    where
        U: Tanh,
    {
        self.activate(|x| x.tanh())
    }

    fn tanh_derivative(&self) -> Self::Cont<U::Output>
    where
        U: Tanh,
    {
        self.activate(|x| x.tanh_derivative())
    }

    fn sigmoid_complex(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.activate(|x| U::one() / (U::one() + (-x).exp()))
    }

    fn sigmoid_complex_derivative(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.activate(|x| {
            let s = U::one() / (U::one() + (-x).exp());
            s * (U::one() - s)
        })
    }

    fn tanh_complex(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.activate(|x| x.tanh())
    }

    fn tanh_complex_derivative(&self) -> Self::Cont<U>
    where
        U: ComplexFloat,
    {
        self.activate(|x| {
            let s = x.tanh();
            U::one() - s * s
        })
    }
}

pub trait NdActivateMut<A, D>
where
    A: ScalarOperand,
    D: Dimension,
{
    type Data: DataMut<Elem = A>;
}
/*
 ************* Implementations *************
*/
impl<U, S> ActivateExt<U> for S where S: Activate<U> {}

impl<A, S, D> Activate<A> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn activate<V, F>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(A) -> V,
    {
        self.mapv(f)
    }
}

impl<'a, A, S, D> Activate<A> for &'a ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<'a, A, S, D> Activate<A> for &'a mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<A, S, D> ActivateMut<A> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}

impl<'a, A, S, D> ActivateMut<A> for &'a mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn activate_inplace<'b, F>(&'b mut self, f: F)
    where
        A: 'b,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
