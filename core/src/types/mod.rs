/*
   Appellation: types <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;
#[cfg(feature = "std")]
pub use self::std_types::*;

pub mod propagate;
pub mod shape;

pub type NdResult<T> = core::result::Result<T, nd::ShapeError>;

#[cfg(feature = "std")]
mod std_types {
    /// A type alias for a boxed [Error](std::error::Error) type that is `Send`, `Sync`, and `'static`.
    pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
    /// A type alias for a boxed [Result](core::result::Result) which returns some object, `T`, and uses a [BoxError] as the error type.
    pub type BoxResult<T = ()> = core::result::Result<T, BoxError>;
}

pub(crate) mod prelude {
    pub use super::NdResult;
    pub use super::propagate::Propagate;
    pub use super::shape::ModelShape;
    #[cfg(feature = "std")]
    pub use super::std_types::*;
}

#[cfg(test)]
mod tests {}
