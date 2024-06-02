/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::rust::Box;

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
impl<U, M> Forward<U> for M
where
    M: Predict<U>,
{
    type Output = M::Output;

    fn forward(&self, args: &U) -> Self::Output {
        self.predict(args).unwrap()
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
#[cfg(any(feature = "alloc", feature = "std"))]
impl<U, V> Predict<U> for Box<dyn Predict<U, Output = V>> {
    type Output = V;

    fn predict(&self, args: &U) -> Result<Self::Output, PredictError> {
        self.as_ref().predict(args)
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
