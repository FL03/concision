/*
    Appellation: precision <module>
    Created At: 2025.11.26:12:09:08
    Contrib: @FL03
*/
use num_traits::Float;

pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

impl<T> FloorDiv for T
where
    T: Copy + core::ops::Div<Output = T> + core::ops::Rem<Output = T> + core::ops::Sub<Output = T>,
{
    type Output = T;

    fn floor_div(self, rhs: Self) -> Self::Output {
        (self - (self % rhs)) / rhs
    }
}

impl<T> RoundTo for T
where
    T: Float,
{
    fn round_to(&self, places: usize) -> Self {
        let factor = T::from(10).unwrap().powi(places as i32);
        (*self * factor).round() / factor
    }
}
