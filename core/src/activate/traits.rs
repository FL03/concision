/*
    Appellation: traits <activate>
    Contrib: @FL03
*/

use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, ScalarOperand};
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

pub trait NdActivate<A, D>
where
    A: ScalarOperand,
    D: Dimension,
{
    type Data: Data<Elem = A>;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B;

    fn linear(&self) -> Array<A::Output, D>
    where
        A: LinearActivation,
    {
        self.activate(|x| x.linear())
    }

    fn linear_derivative(&self) -> Array<A::Output, D>
    where
        A: LinearActivation,
    {
        self.activate(|x| x.linear_derivative())
    }

    fn heavyside(&self) -> Array<A::Output, D>
    where
        A: Heavyside,
    {
        self.activate(|x| x.heavyside())
    }

    fn heavyside_derivative(&self) -> Array<A::Output, D>
    where
        A: Heavyside,
    {
        self.activate(|x| x.heavyside_derivative())
    }

    fn relu(&self) -> Array<A::Output, D>
    where
        A: ReLU,
    {
        self.activate(|x| x.relu())
    }

    fn relu_derivative(&self) -> Array<A::Output, D>
    where
        A: ReLU,
    {
        self.activate(|x| x.relu_derivative())
    }
    ///
    fn sigmoid(&self) -> Array<A::Output, D>
    where
        A: Sigmoid,
    {
        self.activate(|x| x.sigmoid())
    }
    ///
    fn sigmoid_derivative(&self) -> Array<A::Output, D>
    where
        A: Sigmoid,
    {
        self.activate(|x| x.sigmoid_derivative())
    }
    /// Softmax activation function
    /// The softmax function is defined as:
    /// $$ \sigma(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} $$
    fn softmax(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        let exp = self.activate(A::exp);
        &exp / exp.sum()
    }

    fn softmax_axis(&self, axis: usize) -> Array<A, D>
    where
        A: ComplexFloat,
        D: RemoveAxis,
    {
        let exp = self.activate(A::exp);
        let axis = Axis(axis);
        &exp / &exp.sum_axis(axis)
    }

    fn tanh(&self) -> Array<A::Output, D>
    where
        A: Tanh,
    {
        self.activate(|x| x.tanh())
    }

    fn tanh_derivative(&self) -> Array<A::Output, D>
    where
        A: Tanh,
    {
        self.activate(|x| x.tanh_derivative())
    }

    fn sigmoid_complex(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(|x| A::one() / (A::one() + (-x).exp()))
    }

    fn sigmoid_complex_derivative(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(|x| {
            let s = A::one() / (A::one() + (-x).exp());
            s * (A::one() - s)
        })
    }

    fn tanh_complex(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(|x| x.tanh())
    }
    fn tanh_complex_derivative(&self) -> Array<A, D>
    where
        A: ComplexFloat,
    {
        self.activate(|x| {
            let s = x.tanh();
            A::one() - s * s
        })
    }
}

pub trait NdActivateMut<A, D>
where
    A: ScalarOperand,
    D: Dimension,
{
    type Data: DataMut<Elem = A>;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A;
}
/*
 ************* Implementations *************
*/

impl<A, S, D> NdActivate<A, D> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Data = S;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<A, S, D> NdActivateMut<A, D> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Data = S;

    fn activate_inplace<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}

impl<'a, A, S, D> NdActivate<A, D> for &'a ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Data = S;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<'a, A, S, D> NdActivate<A, D> for &'a mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Data = S;

    fn activate<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<'a, A, S, D> NdActivateMut<A, D> for &'a mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Data = S;

    fn activate_inplace<'b, F>(&'b mut self, f: F)
    where
        A: 'b,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
