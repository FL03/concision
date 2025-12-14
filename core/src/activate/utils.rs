/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, ScalarOperand};
use num_traits::{Float, One, Zero};

/// the relu activation function:
///
/// ```math
/// \mbox{f}(x) = \max(0, x)
/// ```
pub fn relu<T>(args: T) -> T
where
    T: PartialOrd + Zero,
{
    if args > T::zero() { args } else { T::zero() }
}

///
///  ```math
/// \frac{df}{dx}=\max(0,1)
/// ```
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
/// the sigmoid activation function:
///
/// ```math
/// f(x)=(1+e^{-x})^{-1}
/// ```
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
/// Softmax function:
///
/// ```math
/// f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
/// ```
pub fn softmax<A, S, D>(args: &ArrayBase<S, D, A>) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    let e = args.exp();
    &e / e.sum()
}
/// Softmax function along a specific axis:
///
/// ```math
/// f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
/// ```
pub fn softmax_axis<A, S, D>(args: &ArrayBase<S, D, A>, axis: usize) -> Array<A, D>
where
    A: Float + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    let axis = Axis(axis);
    let e = args.exp();
    &e / &e.sum_axis(axis)
}
/// Hyperbolic tangent
///
/// ```math
/// f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
/// ```
pub fn tanh<T>(args: T) -> T
where
    T: Float,
{
    args.tanh()
}
/// the derivative of the tanh function
pub fn tanh_derivative<T>(args: T) -> T
where
    T: Float,
{
    let t = tanh(args);
    T::one() - t * t
}

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
/// H(x)=\begin{cases}1 &x\gt{0} \\ 0 &x\leq{0} \end{cases}
/// ```
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() { T::one() } else { T::zero() }
}
