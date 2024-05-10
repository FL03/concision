/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait State {
    type Mode: ParamMode;
}

pub trait ParamMode: 'static {
    const BIASED: bool = false;

    fn is_biased(&self) -> bool {
        Self::BIASED
    }

    private!();
}

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

param_mode! {
    Biased: true,
    Unbiased: false,
}
