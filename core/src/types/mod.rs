/*
   Appellation: types <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;
#[cfg(feature = "std")]
pub use self::std_types::*;

pub mod direction;

/// A type alias for a [Result](core::result::Result) with the crate's [Error](crate::error::Error) type.
/// Defaults to `Result<(), Error>`
pub type Result<T = ()> = core::result::Result<T, crate::error::ErrorKind>;

#[cfg(feature = "std")]
mod std_types {
    ///
    pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
    ///
    pub type BoxResult<T = ()> = core::result::Result<T, BoxError>;
}

pub(crate) mod prelude {
    pub use super::direction::Direction;
    #[cfg(feature = "std")]
    pub use super::std_types::*;
    pub use super::Result;
}

#[cfg(test)]
mod tests {}
