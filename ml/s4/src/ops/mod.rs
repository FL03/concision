/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{convolve::*, discretize::*, gen::*, scan::*};

pub(crate) mod convolve;
pub(crate) mod discretize;
pub(crate) mod gen;
pub(crate) mod scan;

#[cfg(test)]
mod tests {}
