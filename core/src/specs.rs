/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{Add, Div, Mul, Sub};

/// [Numerical] is a basic trait describing numerical objects
pub trait Numerical:
    Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Clone
    + Copy
    + Sized
{
}

impl Numerical for f32 {}

impl Numerical for f64 {}

impl Numerical for i64 {}

impl Numerical for usize {}
