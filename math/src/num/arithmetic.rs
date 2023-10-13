/*
    Appellation: arithmetic <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{self, Add, Div, Mul, Neg, Sub};

pub trait Addition<T>: Add<Output = T> + ops::AddAssign + Sized {}

impl<T> Addition<T> for T where T: Add<Output = Self> + ops::AddAssign + Sized {}

pub trait Division: Div<Output = Self> + ops::DivAssign + Sized {}

impl<T> Division for T where T: Div<Output = Self> + ops::DivAssign + Sized {}

pub trait Multiplication: Mul<Output = Self> + ops::MulAssign + Sized {}

impl<T> Multiplication for T where T: Mul<Output = Self> + ops::MulAssign + Sized {}

pub trait Subtraction: Sub<Output = Self> + ops::SubAssign + Sized {}

impl<T> Subtraction for T where T: Sub<Output = Self> + ops::SubAssign + Sized {}

pub trait Negative: Neg<Output = Self> + Sized {}
