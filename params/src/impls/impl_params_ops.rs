/*
    Appellation: impl_ops <module>
    Contrib: @FL03
*/
use crate::{Params, ParamsBase};
use concision_traits::{ApplyGradient, ApplyGradientExt, Backward, Forward, Norm, Biased, Weighted, Result};
use ndarray::linalg::Dot;
use ndarray::{RawData, prelude::*};
use ndarray::{ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};


impl<A, S, D> ParamsBase<S, D, A>
where
    A: ScalarOperand + Float + FromPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    /// Returns the L1 norm of the parameters (bias and weights).
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

    /// a convenience method used to apply a gradient to the parameters using the given
    /// learning rate.
    pub fn apply_gradient<Delta, Z>(&mut self, grad: &Delta, lr: A) -> Result<Z>
    where
        S: DataMut,
        Self: ApplyGradient<Delta, A, Output = Z>,
    {
        <Self as ApplyGradient<Delta, A>>::apply_gradient(self, grad, lr)
    }

    pub fn apply_gradient_with_decay<Grad, Z>(
        &mut self,
        grad: &Grad,
        lr: A,
        decay: A,
    ) -> Result<Z>
    where
        S: DataMut,
        Self: ApplyGradient<Grad, A, Output = Z>,
    {
        <Self as ApplyGradient<Grad, A>>::apply_gradient_with_decay(self, grad, lr, decay)
    }

    pub fn apply_gradient_with_momentum<Grad, V, Z>(
        &mut self,
        grad: &Grad,
        lr: A,
        momentum: A,
        velocity: &mut V,
    ) -> Result<Z>
    where
        S: DataMut,
        Self: ApplyGradientExt<Grad, A, Output = Z, Velocity = V>,
    {
        <Self as ApplyGradientExt<Grad, A>>::apply_gradient_with_momentum(
            self, grad, lr, momentum, velocity,
        )
    }

    pub fn apply_gradient_with_decay_and_momentum<Grad, V, Z>(
        &mut self,
        grad: &Grad,
        lr: A,
        decay: A,
        momentum: A,
        velocity: &mut V,
    ) -> Result<Z>
    where
        S: DataMut,
        Self: ApplyGradientExt<Grad, A, Output = Z, Velocity = V>,
    {
        <Self as ApplyGradientExt<Grad, A>>::apply_gradient_with_decay_and_momentum(
            self, grad, lr, decay, momentum, velocity,
        )
    }
}

impl<A, S, D> Weighted<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn weights(&self) -> &ArrayBase<S, D, A> {
        self.weights()
    }

    fn weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.weights_mut()
    }
}

impl<A, S, D> Biased<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn bias(&self) -> &ArrayBase<S, D::Smaller, A> {
        self.bias()
    }

    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A> {
        self.bias_mut()
    }
}

impl<A, S, T, D> ApplyGradient<ParamsBase<T, D>, A> for ParamsBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: DataMut<Elem = A>,
    T: Data<Elem = A>,
    D: Dimension,
{
    type Output = ();

    fn apply_gradient(&mut self, grad: &ParamsBase<T, D>, lr: A) -> Result<Self::Output> {
        // apply the bias gradient
        self.bias_mut().apply_gradient(grad.bias(), lr)?;
        // apply the weight gradient
        self.weights_mut().apply_gradient(grad.weights(), lr)?;
        Ok(())
    }

    fn apply_gradient_with_decay(
        &mut self,
        grad: &ParamsBase<T, D>,
        lr: A,
        decay: A,
    ) -> Result<Self::Output> {
        // apply the bias gradient
        self.bias_mut()
            .apply_gradient_with_decay(grad.bias(), lr, decay)?;
        // apply the weight gradient
        self.weights_mut()
            .apply_gradient_with_decay(grad.weights(), lr, decay)?;
        Ok(())
    }
}

impl<A, S, T, D> ApplyGradientExt<ParamsBase<T, D>, A> for ParamsBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: DataMut<Elem = A>,
    T: Data<Elem = A>,
    D: Dimension,
{
    type Velocity = Params<A, D>;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &ParamsBase<T, D>,
        lr: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> Result<()> {
        // apply the bias gradient
        self.bias_mut().apply_gradient_with_momentum(
            grad.bias(),
            lr,
            momentum,
            velocity.bias_mut(),
        )?;
        // apply the weight gradient
        self.weights_mut().apply_gradient_with_momentum(
            grad.weights(),
            lr,
            momentum,
            velocity.weights_mut(),
        )?;
        Ok(())
    }

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &ParamsBase<T, D>,
        lr: A,
        decay: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> Result<()> {
        // apply the bias gradient
        self.bias_mut().apply_gradient_with_decay_and_momentum(
            grad.bias(),
            lr,
            decay,
            momentum,
            velocity.bias_mut(),
        )?;
        // apply the weight gradient
        self.weights_mut().apply_gradient_with_decay_and_momentum(
            grad.weights(),
            lr,
            decay,
            momentum,
            velocity.weights_mut(),
        )?;
        Ok(())
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix2>, ArrayBase<T, Ix1>> for Params<A, Ix1>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        delta: &ArrayBase<T, Ix1>,
        gamma: Self::Elem,
    ) -> Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = delta.t().dot(input);
        // update the weights and bias
        self.weights_mut().apply_gradient(&weight_delta, gamma)?;
        self.bias_mut()
            .apply_gradient(&delta.sum_axis(Axis(0)), gamma)?;
        // return the sum of the squared delta
        Ok(delta.pow2().sum())
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix1>, ArrayBase<T, Ix0>> for Params<A, Ix1>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        delta: &ArrayBase<T, Ix0>,
        gamma: Self::Elem,
    ) -> Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = input * delta;
        // update the weights and bias
        self.weights_mut().apply_gradient(&weight_delta, gamma)?;
        self.bias_mut().apply_gradient(delta, gamma)?;
        // return the sum of the squared delta
        Ok(delta.pow2().sum())
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix1>, ArrayBase<T, Ix1>> for Params<A, Ix2>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        delta: &ArrayBase<T, Ix1>,
        gamma: Self::Elem,
    ) -> Result<Self::Output> {
        // compute the weight gradient
        let dw = &self.weights * delta.t().dot(input);
        // update the weights and bias
        self.weights_mut().apply_gradient(&dw, gamma)?;
        self.bias_mut().apply_gradient(delta, gamma)?;
        // return the sum of the squared delta
        Ok(delta.pow2().sum())
    }
}

impl<A, S, T> Backward<ArrayBase<S, Ix2>, ArrayBase<T, Ix2>> for Params<A, Ix2>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: Data<Elem = A>,
    T: Data<Elem = A>,
{
    type Elem = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        delta: &ArrayBase<T, Ix2>,
        gamma: Self::Elem,
    ) -> Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = input.dot(&delta.t());
        // compute the bias gradient
        let bias_delta = delta.sum_axis(Axis(0));

        self.weights_mut().apply_gradient(&weight_delta, gamma)?;
        self.bias_mut().apply_gradient(&bias_delta, gamma)?;
        // return the sum of the squared delta
        Ok(delta.pow2().sum())
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

    fn forward(&self, input: &X) -> Result<Self::Output> {
        let output = input.dot(&self.weights) + &self.bias;
        Ok(output)
    }
}
