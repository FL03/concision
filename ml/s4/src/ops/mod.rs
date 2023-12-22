/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{convolve::*, discretize::*, gen::*};

pub(crate) mod convolve;
pub(crate) mod discretize;
pub(crate) mod gen;

#[cfg(test)]
mod tests {}
