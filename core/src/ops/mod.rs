/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Overloadable Operations
pub use self::pad::*;

pub(crate) mod pad;

pub(crate) mod prelude {
    pub use super::pad::*;
}

#[cfg(test)]
mod tests {}
