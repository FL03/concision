/*
    Appellation: activate <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! activator {
    ($name:ident::<$out:ty>($rho:expr) $($rest:tt)*) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub struct $name;

        impl<T> $crate::func::activate::Activate<T> for $name $($rest)* {
            type Output = $out;

            fn activate(&self, args: &T) -> Self::Output {
                $rho(args)
            }
        }
    };
}

macro_rules! losses {
    (impl<$($T:ident),* $(,)?> $name:ident::<$lhs:ty, $rhs:ty, Output = $out:ty>($loss:expr) $($rest:tt)*) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub struct $name;

        impl<$($T),*> $crate::func::Loss<$lhs, $rhs> for $name $($rest)* {
            type Output = $out;

            fn loss(&self, a: &$lhs, b: &$rhs) -> Self::Output {
                $loss(a, b)
            }
        }
    };
}
