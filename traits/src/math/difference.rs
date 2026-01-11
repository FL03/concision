/*
    Appellation: difference <module>
    Created At: 2025.11.26:12:09:51
    Contrib: @FL03
*/

/// Compute the percentage difference between two values.
/// The percentage difference is defined as:
///
/// ```math
/// \text{PercentDifference}(x, y) = 2\cdot\frac{|x - y|}{|x| + |y|}
/// ```
pub trait PercentDiff<Rhs = Self> {
    type Output;

    fn percent_diff(self, rhs: Rhs) -> Self::Output;
}

pub trait PercentChange<Rhs = Self> {
    type Output;

    fn percent_change(self, rhs: Rhs) -> Self::Output;
}

/*
 ************* Implementations *************
*/
use num_traits::{FromPrimitive, NumOps, Signed, Zero};

impl<T> PercentDiff for T
where
    T: Copy + Signed + Zero + FromPrimitive + NumOps<T, T>,
{
    type Output = T;

    fn percent_diff(self, rhs: T) -> Self::Output {
        T::from_u8(2).unwrap() * (self - rhs).abs() / (self.abs() + rhs.abs())
    }
}

impl<A, B, C> PercentChange<B> for A
where
    C: core::ops::Div<B, Output = C>,
    for<'b> A: core::ops::Sub<&'b B, Output = C>,
{
    type Output = C;

    fn percent_change(self, rhs: B) -> Self::Output {
        (self - &rhs) / rhs
    }
}

// macro_rules! impl_percent_change {
//     ($($T:ty),* $(,)?) => {
//         $(impl_percent_change! { @impl $T })*
//     };
//     (@impl $T:ty) => {
//         impl PercentChange<$T> for $T {
//             type Output = $T;

//             fn percent_change(self, rhs: $T) -> Self::Output {
//                 (self - rhs) / rhs
//             }
//         }
//     };
// }
