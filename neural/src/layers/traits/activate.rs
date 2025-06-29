/*
    appellation: activate <module>
    authors: @FL03
*/
/// The [`Activator`] trait defines a method for applying an activation function to an input
/// tensor.
pub trait Activator<T> {
    type Output;

    /// Applies the activation function to the input tensor.
    fn activate(&self, input: T) -> Self::Output;
}
/// The [`ActivatorGradient`] trait extends the [`Activator`] trait to include a method for
/// computing the gradient of the activation function.
pub trait ActivatorGradient<T>: Activator<T> {
    type Input;
    type Delta;

    /// compute the gradient of some input
    fn activate_gradient(&self, input: Self::Input) -> Self::Delta;
}

/*
 ************* Implementations *************
*/

impl<X, Y, F> Activator<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}

#[cfg(feature = "alloc")]
mod impl_alloc {
    use super::Activator;
    use alloc::boxed::Box;

    impl<X, Y> Activator<X> for Box<dyn Activator<X, Output = Y>> {
        type Output = Y;

        fn activate(&self, rhs: X) -> Self::Output {
            self.as_ref().rho(rhs)
        }
    }
}

/*
 ************* Implementations *************
*/
macro_rules! activator {
    (@impl $vis:vis struct $name:ident::<$($trait:ident)::*>($method:ident) ) => {

        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        $vis struct $name;

        impl<U> Activator<U> for $name
        where
            U: $($trait)::*,
        {
            type Output = U::Output;

            fn activate(&self, x: U) -> Self::Output {
                x.$method()
            }
        }

        paste::paste! {
            impl<U> ActivatorGradient<U> for $name
            where
                U: $($trait)::*,
            {
                type Input = U;
                type Delta = U::Output;

                fn activate_gradient(&self, inputs: U) -> Self::Delta {
                    inputs.[<$method _derivative>]()
                }
            }
        }
    };
    ($(
        $vis:vis struct $name:ident::<$($trait:ident)::*>($method:ident)
    );* $(;)?) => {
        $(
            activator!(@impl $vis struct $name::<$($trait)::*>($method));
        )*
    };
}

activator! {
    pub struct Linear::<cnc::LinearActivation>(linear);
    pub struct ReLU::<cnc::ReLU>(relu);
    pub struct Sigmoid::<cnc::Sigmoid>(sigmoid);
    pub struct Tanh::<cnc::Tanh>(tanh);
}
