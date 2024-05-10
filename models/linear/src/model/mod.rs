/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::layout::prelude::*;
pub use self::{config::Config, linear::Linear};

mod linear;

pub mod config;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub enum Biased {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub enum Unbiased {}

pub trait ParamMode: 'static {
    type Mode;

    private!();
}

macro_rules! impl_param_ty {
    ($($T:ty),* $(,)?) => {
        $(impl_param_ty!(@impl $T);)*
    };
    (@impl $T:ty) => {
        impl ParamMode for $T {
            type Mode = $T;

            seal!();
        }
    };

}

impl_param_ty!(Biased, Unbiased,);

pub mod layout {
    pub use self::{features::*, layout::*};

    mod features;
    mod layout;

    pub(crate) mod prelude {
        pub use super::features::Features;
        pub use super::layout::Layout;
    }
}

pub(crate) mod prelude {
    pub use super::linear::Linear;
}
