/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Operations
pub use self::{kinds::*, pad::*};

pub(crate) mod kinds;
pub(crate) mod pad;

pub mod fft;

pub trait Operation<T> {
    type Output;

    fn eval(&self, args: &T) -> Self::Output;
}

impl<F, S, T> Operation<T> for F
where
    F: Fn(&T) -> S,
{
    type Output = S;

    fn eval(&self, args: &T) -> Self::Output {
        self(args)
    }
}

#[cfg(test)]
mod tests {}
