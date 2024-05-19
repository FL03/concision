/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;

/// [Forward] describes an object capable of forward propagation.
pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

/// [ForwardIter] describes any iterators whose elements implement [Forward].
/// This trait is typically used in deep neural networks who need to forward propagate
/// across a number of layers.
pub trait ForwardIter<T> {
    type Item: Forward<T, Output = T>;

    fn forward_iter(self, args: &T) -> <Self::Item as Forward<T>>::Output;
}

pub trait Predict<T> {
    type Output;

    fn predict(&self, args: &T) -> Result<Self::Output, PredictError>;
}

/*
 ********* Implementations *********
*/
impl<X, Y, S> Forward<X> for S
where
    S: Predict<X, Output = Y>,
{
    type Output = Y;

    fn forward(&self, args: &X) -> Self::Output {
        if let Ok(y) = self.predict(args) {
            y
        } else {
            panic!("Error in forward propagation")
        }
    }
}

impl<I, M, T> ForwardIter<T> for I
where
    I: IntoIterator<Item = M>,
    M: Forward<T, Output = T>,
    T: Clone,
{
    type Item = M;

    fn forward_iter(self, args: &T) -> M::Output {
        self.into_iter()
            .fold(args.clone(), |acc, m| m.forward(&acc))
    }
}

impl<S, T> Predict<T> for Option<S>
where
    S: Predict<T, Output = T>,
    T: Clone,
{
    type Output = T;

    fn predict(&self, args: &T) -> Result<Self::Output, PredictError> {
        match self {
            Some(s) => s.predict(args),
            None => Ok(args.clone()),
        }
    }
}
