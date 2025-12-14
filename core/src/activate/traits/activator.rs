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
pub trait ActivatorGradient<T> {
    type Rel: Activator<T>;
    type Delta;

    /// compute the gradient of some input
    fn activate_gradient(&self, input: T) -> Self::Delta;
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
impl<X, Y> Activator<X> for alloc::boxed::Box<dyn Activator<X, Output = Y>> {
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self.as_ref().activate(rhs)
    }
}

/*
 ************* Implementations *************
*/
macro_rules! activator {
    ($(
        $vis:vis struct $name:ident.$method:ident where $T:ident: $($trait:ident)::*
    );* $(;)?) => {
        $(
            activator!(@impl $vis struct $name.$method where $T: $($trait)::* );
        )*
    };
    (@impl $vis:vis struct $name:ident.$method:ident where $T:ident: $($trait:ident)::* ) => {

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
                type Rel = Self;
                type Delta = U::Output;

                fn activate_gradient(&self, inputs: U) -> Self::Delta {
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
    pub struct HyperbolicTangent.tanh where T: crate::activate::TanhActivation;
    pub struct HeavySide.heavyside where T: crate::activate::HeavysideActivation;
    pub struct Softmax.softmax where T: crate::activate::SoftmaxActivation;
}
