/*
    Appellation: init <module>
    Contrib: @FL03
*/

/// A trait for creating custom initialization routines for models or other entities.
pub trait InitWith<F, U> {
    type Cont<T>;
    /// consumes the current instance to initialize a new one
    fn init_with(f: F) -> Self::Cont<U>
    where
        F: FnOnce() -> U,
        Self: Sized;
}

#[cfg(feature = "rand")]
/// The [`InitRand`] trait provides a generic interface for initializing objects using
/// random number generators. This trait is particularly useful for types that require
/// random initialization, such as neural network weights, biases, or other parameters.
pub trait InitRand<R: rand::RngCore> {
    type Output;
    /// use the provided random number generator `rng` to initialize the object
    fn init_random(rng: &mut R) -> Self::Output;
}
