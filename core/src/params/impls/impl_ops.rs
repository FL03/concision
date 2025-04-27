/*
    Appellation: impl_ops <module>
    Contrib: @FL03
*/
use crate::params::{Params, ParamsBase};
use crate::traits::{ApplyGradient, ApplyGradientExt, Backward, Forward, Norm};
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> ParamsBase<S, D>
where
    A: ScalarOperand + Float + FromPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    /// Returns the L1 norm of the parameters (bias and weights).
    pub fn l1_norm(&self) -> A {
        let bias = self.bias.l1_norm();
        let weights = self.weights.l1_norm();
        bias + weights
    }
    /// Returns the L2 norm of the parameters (bias and weights).
    pub fn l2_norm(&self) -> A {
        let bias = self.bias.l2_norm();
        let weights = self.weights.l2_norm();
        bias + weights
    }
    /// a convenience method used to apply a gradient to the parameters using the given
    /// learning rate.
    pub fn apply_gradient<Delta, Z>(&mut self, grad: &Delta, lr: A) -> crate::Result<Z>
    where
        S: DataMut,
        Self: ApplyGradient<Delta, Elem = A, Output = Z>,
    {
        <Self as ApplyGradient<Delta>>::apply_gradient(self, grad, lr)
    }

    pub fn apply_gradient_with_decay<Grad, Z>(
        &mut self,
        grad: &Grad,
        lr: A,
        decay: A,
    ) -> crate::Result<Z>
    where
        S: DataMut,
        Self: ApplyGradient<Grad, Elem = A, Output = Z>,
    {
        <Self as ApplyGradient<Grad>>::apply_gradient_with_decay(self, grad, lr, decay)
    }

    pub fn apply_gradient_with_momentum<Grad, V, Z>(
        &mut self,
        grad: &Grad,
        lr: A,
        momentum: A,
        velocity: &mut V,
    ) -> crate::Result<Z>
    where
        S: DataMut,
        Self: ApplyGradientExt<Grad, Elem = A, Output = Z, Velocity = V>,
    {
        <Self as ApplyGradientExt<Grad>>::apply_gradient_with_momentum(
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
    ) -> crate::Result<Z>
    where
        S: DataMut,
        Self: ApplyGradientExt<Grad, Elem = A, Output = Z, Velocity = V>,
    {
        <Self as ApplyGradientExt<Grad>>::apply_gradient_with_decay_and_momentum(
            self, grad, lr, decay, momentum, velocity,
        )
    }
}

impl<A, S, T, D> ApplyGradient<ParamsBase<T, D>> for ParamsBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: DataMut<Elem = A>,
    T: Data<Elem = A>,
    D: Dimension,
{
    type Elem = A;
    type Output = ();

    fn apply_gradient(&mut self, grad: &ParamsBase<T, D>, lr: A) -> crate::Result<Self::Output> {
        // apply the bias gradient
        self.bias.apply_gradient(grad.bias(), lr)?;
        // apply the weight gradient
        self.weights.apply_gradient(grad.weights(), lr)?;
        Ok(())
    }

    fn apply_gradient_with_decay(
        &mut self,
        grad: &ParamsBase<T, D>,
        lr: A,
        decay: A,
    ) -> crate::Result<Self::Output> {
        // apply the bias gradient
        self.bias
            .apply_gradient_with_decay(grad.bias(), lr, decay)?;
        // apply the weight gradient
        self.weights
            .apply_gradient_with_decay(grad.weights(), lr, decay)?;
        Ok(())
    }
}

impl<A, S, T, D> ApplyGradientExt<ParamsBase<T, D>> for ParamsBase<S, D>
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
    ) -> crate::Result<()> {
        // apply the bias gradient
        self.bias
            .apply_gradient_with_momentum(grad.bias(), lr, momentum, velocity.bias_mut())?;
        // apply the weight gradient
        self.weights.apply_gradient_with_momentum(
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
    ) -> crate::Result<()> {
        // apply the bias gradient
        self.bias.apply_gradient_with_decay_and_momentum(
            grad.bias(),
            lr,
            decay,
            momentum,
            velocity.bias_mut(),
        )?;
        // apply the weight gradient
        self.weights.apply_gradient_with_decay_and_momentum(
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
    type HParam = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        delta: &ArrayBase<T, Ix1>,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = delta.t().dot(input);
        // update the weights and bias
        self.weights.apply_gradient(&weight_delta, gamma)?;
        self.bias.apply_gradient(&delta.sum_axis(Axis(0)), gamma)?;
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
    type HParam = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        delta: &ArrayBase<T, Ix0>,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = input * delta;
        // update the weights and bias
        self.weights.apply_gradient(&weight_delta, gamma)?;
        self.bias.apply_gradient(&delta, gamma)?;
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
    type HParam = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix1>,
        delta: &ArrayBase<T, Ix1>,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output> {
        // compute the weight gradient
        let dw = &self.weights * delta.t().dot(input);
        // update the weights and bias
        self.weights.apply_gradient(&dw, gamma)?;
        self.bias.apply_gradient(&delta, gamma)?;
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
    type HParam = A;
    type Output = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        delta: &ArrayBase<T, Ix2>,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output> {
        // compute the weight gradient
        let weight_delta = input.dot(&delta.t());
        // compute the bias gradient
        let bias_delta = delta.sum_axis(Axis(0));

        self.weights.apply_gradient(&weight_delta, gamma)?;
        self.bias.apply_gradient(&bias_delta, gamma)?;
        // return the sum of the squared delta
        Ok(delta.pow2().sum())
    }
}

impl<A, X, Y, Z, S, D> Forward<X> for ParamsBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> X: Dot<ArrayBase<S, D>, Output = Y>,
    Y: for<'a> core::ops::Add<&'a ArrayBase<S, D::Smaller>, Output = Z>,
{
    type Output = Z;

    fn forward(&self, input: &X) -> crate::Result<Self::Output> {
        let output = input.dot(&self.weights) + &self.bias;
        Ok(output)
    }
}

#[cfg(feature = "rand")]
impl<A, S, D> crate::init::Initialize<S, D> for ParamsBase<S, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData<Elem = A>,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: rand_distr::Distribution<A>,
        Sh: ndarray::ShapeBuilder<Dim = D>,
        S: ndarray::DataOwned,
    {
        use rand::SeedableRng;
        Self::rand_with(
            shape,
            distr,
            &mut rand::rngs::SmallRng::from_rng(&mut rand::rng()),
        )
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: rand::Rng + ?Sized,
        Ds: rand_distr::Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: ndarray::DataOwned,
    {
        let shape = shape.into_shape_with_order();
        let bias_shape = shape.raw_dim().remove_axis(Axis(0));
        let bias = ArrayBase::from_shape_fn(bias_shape, |_| distr.sample(rng));
        let weights = ArrayBase::from_shape_fn(shape, |_| distr.sample(rng));
        Self { bias, weights }
    }
}
