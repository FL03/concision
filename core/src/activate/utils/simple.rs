/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num_traits::{One, Zero};

/// the [`linear`] method is essentially a _passthrough_ method often used in simple models
/// or layers where no activation is needed.
pub const fn linear<T>(x: T) -> T {
    x
}

/// the [`linear_derivative`] method always returns `1` as it is a simple, single variable
/// function
pub fn linear_derivative<T>() -> T
where
    T: One,
{
    <T>::one()
}

/// Heaviside activation function:
///
/// ```math
/// H(x) =
/// \left\{
/// \begin{array}{rcl}
/// 1 & \mbox{if} & x\gt{0} \\
/// 0 & \mbox{if} & x\leq{0}
/// \end{array}
/// \right.
/// ```
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() { T::one() } else { T::zero() }
}
