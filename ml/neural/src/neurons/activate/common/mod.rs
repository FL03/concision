/*
    Appellation: common <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # common
pub use self::{binary::*, linear::*, nonlinear::*, nonlinear::*, utils::*};

pub(crate) mod binary;
pub(crate) mod linear;
pub(crate) mod nonlinear;

pub(crate) mod utils {}
