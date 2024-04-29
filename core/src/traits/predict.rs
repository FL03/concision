/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

/// [Backward] describes an object capable of backward propagation.
///
///  
pub trait Backward {
    type Output;

    fn backward(&self) -> Self::Output;
}

pub trait Module<T>: Forward<T> + Backward {
    type Config;
    type Params;

    fn new(config: Self::Config) -> Self;

    fn config(&self) -> &Self::Config;

    fn config_mut(&mut self) -> &mut Self::Config;

    fn parameters(&self) -> Self::Params;
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
    use super::*;

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
}
