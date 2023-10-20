/*
    Appellation: num <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arithmetic::*, utils::*};

pub(crate) mod arithmetic;

pub mod complex;

use std::ops::{self, Div, Mul, Sub};

pub trait Binary: num::One + num::Zero {}

impl<T> Binary for T where T: num::One + num::Zero {}

pub trait Arithmetic:
    ops::Add<Output = Self> + Div<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Sized
{
}

impl<T> Arithmetic for T where
    T: ops::Add<Output = Self>
        + Div<Output = Self>
        + Mul<Output = Self>
        + Sub<Output = Self>
        + Sized
{
}

/// [Numerical] is a basic trait describing numerical objects
pub trait Numerical: Arithmetic + Copy + PartialEq {}

impl<T> Numerical for T where T: Arithmetic + Copy + PartialEq {}

pub(crate) mod utils {}
