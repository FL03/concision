/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

/// Collection of constants used throughout the system
pub(crate) mod constants {}

/// Collection of static references used throughout
pub(crate) mod statics {}

/// Collection of types used throughout the system
pub(crate) mod types {
    ///
    pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
    ///
    pub type BoxResult<T = ()> = std::result::Result<T, BoxError>;
}
