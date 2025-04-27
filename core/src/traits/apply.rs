/*
    Appellation: train <module>
    Contrib: @FL03
*/
/// A trait declaring basic gradient-related routines for a neural network
pub trait ApplyGradient<Delta> {
    type Elem;
    type Output;

    fn apply_gradient(&mut self, grad: &Delta, lr: Self::Elem) -> crate::Result<Self::Output>;

    fn apply_gradient_with_decay(
        &mut self,
        grad: &Delta,
        lr: Self::Elem,
        decay: Self::Elem,
    ) -> crate::Result<Self::Output>;
}

/// This trait extends the [ApplyGradient] trait by allowing for momentum-based optimization
pub trait ApplyGradientExt<Delta>: ApplyGradient<Delta> {
    type Velocity;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &Delta,
        lr: Self::Elem,
        momentum: Self::Elem,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output>;

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &Delta,
        lr: Self::Elem,
        decay: Self::Elem,
        momentum: Self::Elem,
        velocity: &mut Self::Velocity,
    ) -> crate::Result<Self::Output>;
}

use ndarray::{ArrayBase, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, T, D> ApplyGradient<ArrayBase<T, D>> for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: ndarray::DataMut<Elem = A>,
    T: ndarray::Data<Elem = A>,
    D: Dimension,
{
    type Elem = A;
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
impl<A, S, T, D> ApplyGradientExt<ArrayBase<T, D>> for ArrayBase<S, D>
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
