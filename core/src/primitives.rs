/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{constants::*, statics::*, types::*};

/// Collection of constants used throughout the system
mod constants {}

/// Collection of static references used throughout
mod statics {}

/// Collection of types used throughout the system
mod types {
    ///
    pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
    ///
    pub type BoxResult<T = ()> = std::result::Result<T, BoxError>;
}
