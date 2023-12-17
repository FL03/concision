/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{convolve::*, discretize::*};

pub(crate) mod convolve;
pub(crate) mod discretize;

#[cfg(test)]
mod tests {}
