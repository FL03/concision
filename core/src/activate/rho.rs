/*
    Appellation: rho <module>
    Created At: 2026.01.13:18:09:21
    Contrib: @FL03
*/
//! this module defines _structural_ implementations of various activation functions

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

macro_rules! activator {
    ($($vis:vis struct $name:ident::<$T:ident>::$method:ident $({where $($where:tt)*})?),* $(,)?) => {
        $(activator! {
            @impl $vis struct $name::<$T>::$method $({where $($where)*})?
        })*
    };
    (@impl $vis:vis struct $name:ident::<$T:ident>::$method:ident $({where $($where:tt)*})? ) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        $vis struct $name;

        impl<$T> Activator<$T> for $name
        $(where $($where)*)?
        {
            type Output = <$T>::Output;

            fn activate(&self, x: $T) -> Self::Output {
                x.$method()
            }
        }

        paste::paste! {
            impl<$T> ActivatorGradient<$T> for $name
            $(where $($where)*)?,
            {
                type Rel = Self;
                type Delta = <$T>::Output;

                fn activate_gradient(&self, inputs: $T) -> Self::Delta {
                    inputs.[<$method _derivative>]()
                }
            }
        }
    };
}

activator! {
    pub struct Linear::<T>::linear { where T: crate::activate::LinearActivation },
    pub struct ReLU::<T>::relu { where T: crate::activate::ReLUActivation },
    pub struct Sigmoid::<T>::sigmoid { where T: crate::activate::SigmoidActivation },
    pub struct HyperbolicTangent::<T>::tanh { where T: crate::activate::TanhActivation },
    pub struct HeavySide::<T>::heavyside { where T: crate::activate::HeavysideActivation },
    pub struct Softmax::<T>::softmax { where T: crate::activate::SoftmaxActivation },
}
