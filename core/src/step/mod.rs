/*
   Appellation: step <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{stepper::*, utils::*};

pub(crate) mod stepper;

pub mod linspace;

pub trait Step {}

pub(crate) mod utils {}
