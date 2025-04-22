/*
    Appellation: init <module>
    Contrib: @FL03
*/

/// The [Init] trait is a consuming initialization method
pub trait Init {
    /// consumes the current instance to initialize a new one
    fn init(self) -> Self
    where
        Self: Sized;
}

/// A trait for initializing an object in-place
pub trait InitInplace {
    /// initialize the object in-place and return a mutable reference to it.
    fn init_inplace(&mut self) -> &mut Self;
}
