/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Operations
pub use self::kinds::*;

pub(crate) mod kinds;

pub trait Operation<T> {
    type Output;

    fn eval(&self, args: &T) -> Self::Output;
}

#[cfg(test)]
mod tests {}
