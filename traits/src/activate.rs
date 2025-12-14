/*
    appellation: unary <module>
    authors: @FL03
*/
pub use self::impl_activator::*;

mod impl_activator;
mod impl_linear;
mod impl_nonlinear;
#[allow(dead_code)]
pub(crate) mod utils;

/// An [`Activator`] defines an interface for _structural_ activation functions that can be
/// applied onto various types.
pub trait Activator<T> {
    type Output;

    /// Applies the activation function to the input tensor.
    fn activate(&self, input: T) -> Self::Output;
}
/// The [`ActivatorGradient`] trait extends the [`Activator`] trait to include a method for
/// computing the gradient of the activation function.
pub trait ActivatorGradient<T> {
    type Rel: Activator<T>;
    type Delta;

    /// compute the gradient of some input
    fn activate_gradient(&self, input: T) -> Self::Delta;
}

macro_rules! unary {
    (@impl $name:ident::$call:ident($($rest:tt)*)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call($($rest)*) -> Self::Output;

                fn [<$call _derivative>]($($rest)*) -> Self::Output;
            }
        }
    };
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };
}

unary! {
    HeavysideActivation::heavyside(self),
    LinearActivation::linear(self),
    SigmoidActivation::sigmoid(self),
    SoftmaxActivation::softmax(&self),
    ReLUActivation::relu(&self),
    TanhActivation::tanh(&self),
}

pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}
