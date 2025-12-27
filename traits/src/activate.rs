/*
    appellation: unary <module>
    authors: @FL03
*/
pub use self::impl_activator::*;

mod impl_activator;
mod impl_linear;
mod impl_nonlinear;
mod impl_rho;

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

use num_traits::{One, Zero};

/// The [`Rho`] trait defines a set of activation functions that can be applied to an
/// implementor of the [`Apply`] trait. It provides methods for common activation functions
/// such as linear, heavyside, ReLU, sigmoid, and tanh, along with their derivatives.
/// The trait is generic over a type `U`, which represents the data type of the input to the
/// activation functions. The trait also inherits a type alias `Cont<U>` to allow for variance
/// w.r.t. the outputs of defined methods.
pub trait Rho<T> {
    type Cont<_V>;

    fn rho<F, V>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(T) -> V;
    /// the linear activation function is essentially a passthrough function, simply cloning
    /// the content.
    fn linear(&self) -> Self::Cont<T> {
        self.rho(|x| x)
    }

    fn linear_derivative(&self) -> Self::Cont<T::Output>
    where
        T: One,
    {
        self.rho(|_| <T>::one())
    }

    fn heavyside(&self) -> Self::Cont<T>
    where
        T: One + Zero + PartialOrd,
    {
        self.rho(|x| if x > T::zero() { T::one() } else { T::zero() })
    }

    fn heavyside_derivative(&self) -> Self::Cont<T::Output>
    where
        T: HeavysideActivation,
    {
        self.rho(|x| x.heavyside_derivative())
    }

    fn relu(&self) -> Self::Cont<T::Output>
    where
        T: ReLUActivation,
    {
        self.rho(|x| x.relu())
    }

    fn relu_derivative(&self) -> Self::Cont<T::Output>
    where
        T: ReLUActivation,
    {
        self.rho(|x| x.relu_derivative())
    }

    fn sigmoid(&self) -> Self::Cont<T::Output>
    where
        T: SigmoidActivation,
    {
        self.rho(|x| x.sigmoid())
    }

    fn sigmoid_derivative(&self) -> Self::Cont<T::Output>
    where
        T: SigmoidActivation,
    {
        self.rho(|x| x.sigmoid_derivative())
    }

    fn tanh(&self) -> Self::Cont<T::Output>
    where
        T: TanhActivation,
    {
        self.rho(|x| x.tanh())
    }

    fn tanh_derivative(&self) -> Self::Cont<T::Output>
    where
        T: TanhActivation,
    {
        self.rho(|x| x.tanh_derivative())
    }
}

/// The [`RhoComplex`] trait is similar to the [`Rho`] trait in that it provides various
/// activation functions for implementos of the [`Apply`] trait, however, instead of being
/// truly generic over a type `U`, it is generic over a type `U` that implements the
/// [`ComplexFloat`] trait. This enables the use of complex numbers in the activation
/// functions, something particularly useful for signal-based workloads.
///
/// **note**: The [`Rho`] and [`RhoComplex`] traits are not intended to be used together, hence
/// why the implemented methods are not given alternative or unique name between the two
/// traits. If you happen to import both within the same file, you will more than likely need
/// to use a fully qualified syntax to disambiguate the two traits. If this becomes a problem,
/// we may consider renaming the _complex_ methods accordingly to differentiate them from the
/// _standard_ methods.
#[cfg(feature = "complex")]
pub trait RhoComplex<U>: Rho<U>
where
    U: num_complex::ComplexFloat,
{
    fn sigmoid(&self) -> Self::Cont<U> {
        self.rho(|x| U::one() / (U::one() + (-x).exp()))
    }

    fn sigmoid_derivative(&self) -> Self::Cont<U> {
        self.rho(|x| {
            let s = U::one() / (U::one() + (-x).exp());
            s * (U::one() - s)
        })
    }

    fn tanh(&self) -> Self::Cont<U> {
        self.rho(|x| x.tanh())
    }

    fn tanh_derivative(&self) -> Self::Cont<U> {
        self.rho(|x| {
            let s = x.tanh();
            U::one() - s * s
        })
    }
}

/*
 ************* Implementations *************
*/
