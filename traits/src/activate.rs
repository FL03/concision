/*
    appellation: unary <module>
    authors: @FL03
*/
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

pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}
pub trait Rho<T> {
    type Cont<U>;

    fn rho<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U;
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

macro_rules! activator {
    ($($vis:vis struct $name:ident.$method:ident where $T:ident: $($trait:ident)::*);* $(;)?) => {
        $(activator! {
            @impl $vis struct $name.$method where $T: $($trait)::*
        })*
    };
    (@impl $vis:vis struct $name:ident.$method:ident where $T:ident: $($trait:ident)::* ) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        $vis struct $name;

        impl<$T> Activator<$T> for $name
        where
            $T: $($trait)::*,
        {
            type Output = <$T>::Output;

            fn activate(&self, x: $T) -> Self::Output {
                x.$method()
            }
        }

        paste::paste! {
            impl<$T> ActivatorGradient<$T> for $name
            where
                $T: $($trait)::*,
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
    pub struct Linear.linear where T: crate::activate::LinearActivation;
    pub struct ReLU.relu where T: crate::activate::ReLUActivation;
    pub struct Sigmoid.sigmoid where T: crate::activate::SigmoidActivation;
    pub struct TanhActivator.tanh where T: crate::activate::TanhActivation;
    pub struct HeavySide.heavyside where T: crate::activate::HeavysideActivation;
    pub struct Softmax.softmax where T: crate::activate::SoftmaxActivation;
}
