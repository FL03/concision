/*
    Appellation: precision <module>
    Created At: 2025.11.26:12:09:08
    Contrib: @FL03
*/
use num_traits::{Float, Num};

/// divide two values and round down to the nearest integer.
fn floor_div<T>(numerator: T, denom: T) -> T
where
    T: Copy + core::ops::Div<Output = T> + core::ops::Rem<Output = T> + core::ops::Sub<Output = T>,
{
    (numerator - (numerator % denom)) / denom
}

/// Round the given value to the given number of decimal places.
fn round_to<T>(val: T, decimals: usize) -> T
where
    T: Float,
{
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

impl<T> FloorDiv for T
where
    T: Copy + Num,
{
    type Output = T;

    fn floor_div(self, rhs: Self) -> Self::Output {
        floor_div(self, rhs)
    }
}

impl<T> RoundTo for T
where
    T: Float,
{
    fn round_to(&self, places: usize) -> Self {
        round_to(*self, places)
    }
}
