/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/


pub trait ParamMode: 'static {
    const BIASED: bool = false;

    fn is_biased(&self) -> bool {
        core::any::TypeId::of::<Self>() == core::any::TypeId::of::<Biased>()
    }

    private!();
}

macro_rules! param_mode {
    {$($T:ident: $opt:expr),* $(,)?} => {
        $(param_mode!(@impl $T: $opt);)*
    };
    (@impl $T:ident: $opt:expr) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
        pub enum $T {}

        impl ParamMode for $T {
            const BIASED: bool = $opt;

            fn is_biased(&self) -> bool {
                $opt
            }

            seal!();
        }
    };

}

param_mode!{
    Biased: true, 
    Unbiased: false,
}
