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

/// Compute the softmax activation along a specified axis.
pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
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

unary! {
    HeavysideActivation::heavyside(self),
    LinearActivation::linear(self),
    SigmoidActivation::sigmoid(self),
    SoftmaxActivation::softmax(&self),
    ReLUActivation::relu(&self),
    TanhActivation::tanh(&self),
}

activator! {
    pub struct Linear::<T>::linear { where T: LinearActivation },
    pub struct ReLU::<T>::relu { where T: ReLUActivation },
    pub struct Sigmoid::<T>::sigmoid { where T: SigmoidActivation },
    pub struct HyperbolicTangent::<T>::tanh { where T: TanhActivation },
    pub struct HeavySide::<T>::heavyside { where T: HeavysideActivation },
    pub struct Softmax::<T>::softmax { where T: SoftmaxActivation },
}
