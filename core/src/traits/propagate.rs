/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;

/// [Backward] describes an object capable of backward propagation.
///
///  
pub trait Backward {
    type Output;

    fn backward(&self) -> Self::Output;
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

// Trait implementations
mod impls {
    use super::{Backward, Forward, ForwardIter};

    impl<S> Backward for Option<S>
    where
        S: Backward,
    {
        type Output = Option<S::Output>;

        fn backward(&self) -> Self::Output {
            match self {
                Some(s) => Some(s.backward()),
                None => None,
            }
        }
    }

    impl<S, T> Forward<T> for Option<S>
    where
        S: Forward<T, Output = T>,
        T: Clone,
    {
        type Output = T;

        fn forward(&self, args: &T) -> Self::Output {
            match self {
                Some(s) => s.forward(args),
                None => args.clone(),
            }
        }
    }

    impl<S, T> Forward<T> for S
    where
        S: AsRef<dyn Forward<T, Output = T>>,
        T: Clone,
    {
        type Output = T;

        fn forward(&self, args: &T) -> Self::Output {
            self.as_ref().forward(args)
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
}
