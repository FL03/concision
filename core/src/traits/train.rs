/*
    Appellation: train <module>
    Contrib: @FL03
*/
/// A trait declaring basic gradient-related routines for a neural network
pub trait ApplyGradient<Grad, A> {
    type Output;

    fn apply_gradient(&mut self, grad: &Grad, lr: A) -> crate::Result<Self::Output>;

    fn apply_gradient_with_decay(
        &mut self,
        grad: &Grad,
        lr: A,
        decay: A,
    ) -> crate::Result<Self::Output>;
}

/// This trait extends the [ApplyGradient] trait by allowing for momentum-based optimization
pub trait ApplyGradientExt<Grad, A>: ApplyGradient<Grad, A> {
    type Velocity;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &Grad,
        lr: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output>;

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &Grad,
        lr: A,
        decay: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output>;
}

/// A simple trait denoting a single backward pass through a layer of a neural network; the
/// trait
pub trait Backward<X, Y> {
    type HParam;
    type Output;

    fn backward(
        &mut self,
        input: &X,
        delta: &Y,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output>;
}

use ndarray::{ArrayBase, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, T, D> ApplyGradient<ArrayBase<T, D>, A> for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: ndarray::DataMut<Elem = A>,
    T: ndarray::Data<Elem = A>,
    D: Dimension,
{
    type Output = ();

    fn apply_gradient(&mut self, grad: &ArrayBase<T, D>, lr: A) -> crate::Result<Self::Output> {
        if self.shape() != grad.shape() {
            return Err(
                ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
            );
        }
        let batch_size = if grad.shape().len() > 0 {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        self.scaled_add(lr / batch_size, &grad);
        Ok(())
    }

    fn apply_gradient_with_decay(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        decay: A,
    ) -> crate::Result<Self::Output> {
        if self.shape() != grad.shape() {
            return Err(
                ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
            );
        }
        let batch_size = if grad.shape().len() > 0 {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        self.scaled_add(lr / batch_size, &(grad + &*self * decay));
        Ok(())
    }
}
impl<A, S, T, D> ApplyGradientExt<ArrayBase<T, D>, A> for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: ndarray::DataMut<Elem = A>,
    T: ndarray::Data<Elem = A>,
    D: Dimension,
{
    type Velocity = ndarray::Array<A, D>;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output> {
        if self.shape() != grad.shape() {
            return Err(
                ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
            );
        }
        let batch_size = if grad.shape().len() > 0 {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        *velocity = &*velocity * momentum + grad * (A::one() - momentum);
        self.scaled_add(lr / batch_size, &velocity);
        Ok(())
    }

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        decay: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output> {
        if self.shape() != grad.shape() {
            return Err(
                ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into(),
            );
        }
        let batch_size = if grad.shape().len() > 0 {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };

        let adjusted_grad = grad + &*self * decay;
        *velocity = &*velocity * momentum + adjusted_grad * (A::one() - momentum);
        self.scaled_add(lr / batch_size, &velocity);
        Ok(())
    }
}
