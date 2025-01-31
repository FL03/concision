/*
    Appellation: activate <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! activator {
    ($name:ident::<$out:ty>($rho:expr) $($rest:tt)*) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub struct $name;

        impl<'a, T> $crate::func::activate::Activate<&'a T> for $name $($rest)* {
            type Output = $out;

            fn activate(&self, args: &'a T) -> Self::Output {
                $rho(args)
            }
        }
    };
}
