/*
    Appellation: impl_nonlinear <module>
    Created At: 2025.12.14:11:13:15
    Contrib: @FL03
*/
use super::{ReLUActivation, SigmoidActivation, SoftmaxActivation, TanhActivation};
use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, One, Zero};

impl<A, S, D> ReLUActivation for ArrayBase<S, D, A>
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

impl<A, S, D> SigmoidActivation for ArrayBase<S, D, A>
where
    A: 'static + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn sigmoid(self) -> Self::Output {
        let dim = self.dim();
        let ones = Array::<A, D>::ones(dim);

        (ones + self.signum().exp()).recip()
    }

    fn sigmoid_derivative(self) -> Self::Output {
        self.mapv(|i| {
            let s = (A::one() + i.neg().exp()).recip();
            s * (A::one() - s)
        })
    }
}

impl<A, S, D> SoftmaxActivation for ArrayBase<S, D, A>
where
    A: ScalarOperand + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn softmax(&self) -> Self::Output {
        let exp = self.exp();
        &exp / exp.sum()
    }

    fn softmax_derivative(&self) -> Self::Output {
        let softmax = self.softmax();

        let ones = Array::<A, D>::ones(self.dim());
        &softmax * (&ones - &softmax)
    }
}

impl<A, S, D> TanhActivation for ArrayBase<S, D, A>
where
    A: 'static + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tanh(&self) -> Self::Output {
        self.mapv(|i| i.tanh())
    }

    fn tanh_derivative(&self) -> Self::Output {
        self.mapv(|i| A::one() - i.tanh().powi(2))
    }
}
