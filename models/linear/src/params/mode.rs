/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::option::Option;

pub trait Toggle: 'static {}

pub trait ParamMode: Toggle {
    const BIASED: bool = false;

    fn is_biased(&self) -> bool {
        core::any::type_name::<Self>() == core::any::type_name::<Biased>()
    }

    private!();
}

/*
 ************* Implementations *************
*/

impl<T> ParamMode for Option<T>
where
    T: 'static,
{
    const BIASED: bool = false;

    fn is_biased(&self) -> bool {
        self.is_some()
    }

    seal!();
}

macro_rules! mode {
    {$($T:ident: $opt:expr),* $(,)?} => {
        $(mode!(@impl $T: $opt);)*
    };
    (@impl $T:ident: $opt:expr) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
        pub enum $T {}

        impl Toggle for $T {}

        impl ParamMode for $T {
            const BIASED: bool = $opt;

            fn is_biased(&self) -> bool {
                $opt
            }

            seal!();
        }
    };

}


macro_rules! impl_toggle {
    ($($scope:ident$(<$T:ident>)?),* $(,)?) => {
        $(impl_toggle!(@impl $scope$(<$T>)?);)*
    };
    (@impl $scope:ident$(<$T:ident>)?) => {
        impl$(<$T>)? Toggle for $scope$(<$T> where $T: 'static)? {}
    };
}

mode! {
    Biased: true,
    Unbiased: false,
}

impl_toggle!(bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, Option<T>);