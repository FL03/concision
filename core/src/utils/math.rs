/*
    Appellation: math <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::traits::{Float, Num};

///
pub fn floor_div<T>(numerator: T, denom: T) -> T
where
    T: Copy + Num,
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
