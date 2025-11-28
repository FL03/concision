/*
    appellation: gradient <module>
    authors: @FL03
*/

/// the [`Gradient`] trait defines the gradient of a function, which is a function that
/// takes an input and returns a delta, which is the change in the output with respect to
/// the input.
pub trait Gradient<S, D> {
    type Elem;
    type Delta<_S, _D>;

    fn grad(&self, rhs: &Self::Delta<S, D>) -> Self::Delta<S, D>;
}

/// A trait declaring basic gradient-related routines for a neural network
pub trait ApplyGradient<Delta, T> {
    type Output;

    fn apply_gradient(&mut self, grad: &Delta, lr: T) -> Option<Self::Output>;

    fn apply_gradient_with_decay(&mut self, grad: &Delta, lr: T, decay: T) -> Option<Self::Output>;
}

/// This trait extends the [ApplyGradient] trait by allowing for momentum-based optimization
pub trait ApplyGradientExt<Delta, T>: ApplyGradient<Delta, T> {
    type Velocity;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &Delta,
        lr: T,
        momentum: T,
        velocity: &mut Self::Velocity,
    ) -> Option<Self::Output>;

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &Delta,
        lr: T,
        decay: T,
        momentum: T,
        velocity: &mut Self::Velocity,
    ) -> Option<Self::Output>;
}

/*
 ************* Implementations *************
*/

use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, T, D> ApplyGradient<ArrayBase<T, D>, A> for ArrayBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: DataMut<Elem = A>,
    T: Data<Elem = A>,
    D: Dimension,
{
    type Output = ();

    fn apply_gradient(&mut self, grad: &ArrayBase<T, D>, lr: A) -> Option<Self::Output> {
        if self.shape() != grad.shape() {
            // return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
            return None;
        }
        let batch_size = if !grad.shape().is_empty() {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        self.scaled_add(lr / batch_size, grad);
        Some(())
    }

    fn apply_gradient_with_decay(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        decay: A,
    ) -> Option<Self::Output> {
        if self.shape() != grad.shape() {
            // return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
            return None;
        }
        let batch_size = if !grad.shape().is_empty() {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        let rhs = grad + &*self * decay;
        self.scaled_add(lr / batch_size, &rhs);
        Some(())
    }
}

impl<A, S, T, D> ApplyGradientExt<ArrayBase<T, D>, A> for ArrayBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    S: DataMut<Elem = A>,
    T: Data<Elem = A>,
    D: Dimension,
{
    type Velocity = Array<A, D>;

    fn apply_gradient_with_momentum(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> Option<Self::Output> {
        if self.shape() != grad.shape() {
            // return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
            return None;
        }
        let batch_size = if !grad.shape().is_empty() {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };
        *velocity = &*velocity * momentum + grad * (A::one() - momentum);
        self.scaled_add(lr / batch_size, velocity);
        Some(())
    }

    fn apply_gradient_with_decay_and_momentum(
        &mut self,
        grad: &ArrayBase<T, D>,
        lr: A,
        decay: A,
        momentum: A,
        velocity: &mut Self::Velocity,
    ) -> Option<Self::Output> {
        if self.shape() != grad.shape() {
            // return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
            return None;
        }
        let batch_size = if !grad.shape().is_empty() {
            A::from_usize(self.shape()[0]).unwrap()
        } else {
            A::one()
        };

        let adjusted_grad = grad + &*self * decay;
        *velocity = &*velocity * momentum + adjusted_grad * (A::one() - momentum);
        self.scaled_add(lr / batch_size, velocity);
        Some(())
    }
}
