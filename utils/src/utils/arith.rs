/*
    Appellation: arith <utils>
    Contrib: @FL03
*/
use num_traits::Float;

/// divide two values and round down to the nearest integer.
pub fn floor_div<T>(numerator: T, denom: T) -> T
where
    T: Copy + core::ops::Div<Output = T> + core::ops::Rem<Output = T> + core::ops::Sub<Output = T>,
{
    (numerator - (numerator % denom)) / denom
}

/// Round the given value to the given number of decimal places.
pub fn round_to<T>(val: T, decimals: usize) -> T
where
    T: Float,
{
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}
