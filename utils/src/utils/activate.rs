/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::{Float, One, Zero};

/// Heaviside activation function
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() { T::one() } else { T::zero() }
}
/// the relu activation function: $f(x) = \max(0, x)$
pub fn relu<T>(args: T) -> T
where
    T: PartialOrd + Zero,
{
    if args > T::zero() { args } else { T::zero() }
}

pub fn relu_derivative<T>(args: T) -> T
where
    T: PartialOrd + One + Zero,
{
    if args > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}
/// the sigmoid activation function: $f(x) = \frac{1}{1 + e^{-x}}$
pub fn sigmoid<T>(args: T) -> T
where
    T: Float,
{
    (T::one() + args.neg().exp()).recip()
}
/// the derivative of the sigmoid function
pub fn sigmoid_derivative<T>(args: T) -> T
where
    T: Float,
{
    let s = sigmoid(args);
    s * (T::one() - s)
}
/// Softmax function: $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
pub fn softmax<A, S, D>(args: &ArrayBase<S, D>) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    let e = args.exp();
    &e / e.sum()
}
/// Softmax function along a specific axis: $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
pub fn softmax_axis<A, S, D>(args: &ArrayBase<S, D>, axis: usize) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    let axis = Axis(axis);
    let e = args.exp();
    &e / &e.sum_axis(axis)
}
/// the tanh activation function: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
pub fn tanh<T>(args: T) -> T
where
    T: num::traits::Float,
{
    args.tanh()
}
/// the derivative of the tanh function
pub fn tanh_derivative<T>(args: T) -> T
where
    T: num::traits::Float,
{
    let t = tanh(args);
    T::one() - t * t
}
