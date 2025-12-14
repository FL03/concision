/*
    appellation: activate <module>
    authors: @FL03
*/
use super::{Activator, ActivatorGradient};

impl<X, Y, F> Activator<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}

// impl<F, S, D, A, B> Activator<ArrayBase<S, D, A>> for F
// where
//     F: Activator<A, Output = B>,
//     S: Data<Elem = A>,
//     D: Dimension,
// {
//     type Output = Array<B, D>;

//     fn activate(&self, rhs: ArrayBase<S, D, A>) -> Self::Output {
//         rhs.mapv(|x| self.activate(x))
//     }
// }

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
