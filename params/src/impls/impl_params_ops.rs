/*
    Appellation: impl_ops <module>
    Contrib: @FL03
*/
use crate::{Params, ParamsBase};
use concision_traits::{Backward, Forward, Norm};
use ndarray::linalg::Dot;
use ndarray::{
    Array, ArrayBase, ArrayView, Data, Dimension, Ix0, Ix1, Ix2, RemoveAxis, ScalarOperand,
};
use num_traits::{Float, FromPrimitive, Num};

impl<A, S, D> ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
{
    /// execute a single backward propagation
    pub fn backward<X, Y>(&mut self, input: &X, grad: &Y, lr: A)
    where
        Self: Backward<X, Y, Elem = A>,
    {
        <Self as Backward<X, Y>>::backward(self, input, grad, lr)
    }
    /// forward propagation
    pub fn forward<X, Y>(&self, input: &X) -> Y
    where
        Self: Forward<X, Output = Y>,
    {
        <Self as Forward<X>>::forward(self, input)
    }
}

impl<A, S, D> ParamsBase<S, D, A>
where
    A: ScalarOperand + Float + FromPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    /// computes the `l1` normalization of the current weights and biases
    pub fn l1_norm(&self) -> A {
        let bias = self.bias().l1_norm();
        let weights = self.weights().l1_norm();
        bias + weights
    }
    /// Returns the L2 norm of the parameters (bias and weights).
    pub fn l2_norm(&self) -> A {
        let bias = self.bias().l2_norm();
        let weights = self.weights().l2_norm();
        bias + weights
    }
}

impl<A, X, Y, Z, S, D> Forward<X> for ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> X: Dot<ArrayBase<S, D>, Output = Y>,
    Y: for<'a> core::ops::Add<&'a ArrayBase<S, D::Smaller>, Output = Z>,
{
    type Output = Z;

    fn forward(&self, input: &X) -> Self::Output {
        input.dot(self.weights()) + self.bias()
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix0, A>, ArrayBase<T, Ix0, A>> for Params<A, Ix1>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix0, A>,
        delta: &ArrayBase<T, Ix0, A>,
        gamma: Self::Elem,
    ) {
        self.weights_mut().scaled_add(gamma, &(input * delta));
        self.bias_mut().scaled_add(gamma, delta);
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix1, A>, ArrayBase<T, Ix1, A>> for Params<A, Ix2>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix1, A>,
        delta: &ArrayBase<T, Ix1, A>,
        gamma: Self::Elem,
    ) {
        self.weights_mut().scaled_add(gamma, &(delta * input));
        self.bias_mut().scaled_add(gamma, delta);
    }
}

impl<A, D1, D2, S1, S2> Backward<ArrayBase<S1, D1, A>, ArrayBase<S2, D2, A>> for Params<A, D1>
where
    A: 'static + Copy + Num,
    D1: RemoveAxis<Smaller = D2>,
    D2: Dimension<Larger = D1>,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    for<'b> &'b ArrayBase<S1, D1, A>: Dot<ArrayView<'b, A, D2>, Output = Array<A, D2>>,
{
    type Elem = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S1, D1, A>,
        delta: &ArrayBase<S2, D2, A>,
        gamma: Self::Elem,
    ) {
        self.weights_mut().backward(input, delta, gamma);
        self.bias_mut().scaled_add(gamma, delta);
    }
}
