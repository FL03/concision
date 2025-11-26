/*
    Appellation: sigmoid <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::activate::{ReLU, Sigmoid, Softmax, TanhActivation, utils::sigmoid_derivative};

use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, One, Zero};

impl<A, S, D> ReLU for ArrayBase<S, D, A>
where
    A: Copy + PartialOrd + Zero + One,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn relu(&self) -> Self::Output {
        self.map(|&i| if i > A::zero() { i } else { A::zero() })
    }

    fn relu_derivative(&self) -> Self::Output {
        self.map(|&i| if i > A::zero() { A::one() } else { A::zero() })
    }
}

impl<A, S, D> Sigmoid for ArrayBase<S, D, A>
where
    A: ScalarOperand + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn sigmoid(self) -> Self::Output {
        let dim = self.dim();
        let ones = Array::<A, D>::ones(dim);

        (ones + self.map(|&i| i.neg().exp())).recip()
    }

    fn sigmoid_derivative(self) -> Self::Output {
        self.mapv(|i| sigmoid_derivative(i))
    }
}

impl<A, S, D> Softmax for ArrayBase<S, D, A>
where
    A: ScalarOperand + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn softmax(&self) -> Self::Output {
        let e = self.exp();
        &e / e.sum()
    }

    fn softmax_derivative(&self) -> Self::Output {
        let e = self.exp();
        let sum = e.sum();
        let softmax = &e / sum;

        let ones = Array::<A, D>::ones(self.dim());
        &softmax * (&ones - &softmax)
    }
}

impl<A, S, D> TanhActivation for ArrayBase<S, D, A>
where
    A: ScalarOperand + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tanh(&self) -> Self::Output {
        self.map(|i| i.tanh())
    }

    fn tanh_derivative(&self) -> Self::Output {
        self.map(|i| A::one() - i.tanh().powi(2))
    }
}
