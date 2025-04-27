/*
    Appellation: init <module>
    Contrib: @FL03
*/

/// A trait for creating custom initialization routines for models or other entities.
pub trait Init {
    /// consumes the current instance to initialize a new one
    fn init(self) -> Self
    where
        Self: Sized;
}

/// This trait enables models to implement custom, in-place initialization methods.
pub trait InitInplace {
    /// initialize the object in-place and return a mutable reference to it.
    fn init_inplace(&mut self) -> &mut Self;
}
