/*
    Appellation: init <module>
    Created At: 2026.01.13:18:38:50
    Contrib: @FL03
*/

/// [`InitWith`] enables a container to
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

/// [`Initialize`] provides a mechanism for _initializing_ some object using a value of type
/// `T` to produce another object.
pub trait Initialize<T> {
    type Output;
    /// initializes the object using the given value, consuming the caller to produce another
    /// object
    fn init(self, with: T) -> Self::Output;
}
