/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::Toggle;
use core::option::Option;

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

mode! {
    Biased: true,
    Unbiased: false,
}

