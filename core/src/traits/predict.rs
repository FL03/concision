/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;

pub trait Activate {
    type Output;

    fn activate(&self) -> Self::Output;
}

pub trait FeedForward: Forward<Self::Params> {
    type Params;
}

/// [Forward] describes an object capable of forward propagation.
pub trait Forward<T> {
    type Output;

    fn forward(&self, args: &T) -> Self::Output;
}

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
impl<S, T> Forward<T> for S
where
    S: Predict<T, Output = T>,
    T: Clone,
{
    type Output = T;

    fn forward(&self, args: &T) -> Self::Output {
        self.predict(args).unwrap()
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

impl<S, T> Predict<T> for S
where
    S: AsRef<dyn Predict<T, Output = T>>,
    T: Clone,
{
    type Output = T;

    fn predict(&self, args: &T) -> Result<Self::Output, PredictError> {
        self.as_ref().predict(args)
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
