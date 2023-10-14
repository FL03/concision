/*
    Appellation: arithmetic <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops;

pub trait Addition<T>: ops::Add<Output = T> + ops::AddAssign<T> + Sized {}

impl<T> Addition<T> for T where T: ops::Add<Output = Self> + ops::AddAssign<T> + Sized {}

pub trait Division<T>: ops::Div<Output = T> + ops::DivAssign<T> + Sized {}

impl<T> Division<T> for T where T: ops::Div<Output = Self> + ops::DivAssign + Sized {}

pub trait Multiplication<T>: ops::Mul<Output = T> + ops::MulAssign<T> + Sized {}

impl<T> Multiplication<T> for T where T: ops::Mul<Output = Self> + ops::MulAssign + Sized {}

pub trait Subtraction<T>: ops::Sub<Output = T> + ops::SubAssign<T> + Sized {}

impl<T> Subtraction<T> for T where T: ops::Sub<Output = Self> + ops::SubAssign + Sized {}
