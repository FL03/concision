/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait RawField {
    private!();
}

macro_rules! impl_raw_private {

    (impl$(<$($T:ident),*>)? $trait:ident for $name:ident$(<$($V:ident),*>)? $(where $($rest:tt)*)?) => {
        impl$(<$($T),*>)? $trait for $name$(<$($V),*>)? $(where $($rest)*)? {
            seal!();
        }
    };
}

macro_rules! toggle {
    (@impl $vis:vis enum $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(
            feature = "serde",
            derive(serde_derive::Deserialize, serde_derive::Serialize),
            serde(rename_all = "lowercase", untagged)
        )]
        pub enum $name {}
    };
    ($vis:vis enum { $($name:ident),* $(,)? }) => {
        $(
            toggle!(@impl $vis enum $name);
            impl_raw_private!(impl RawField for $name);
        )*
    };
}

toggle! {
    pub enum {
        C,
        R
    }
}

///
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    scsys::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantArray,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Deserialize, serde_derive::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[strum(serialize_all = "lowercase")]
pub enum FftMode {
    #[default]
    Complex,
    Real,
}

///
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    scsys::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantArray,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Deserialize, serde_derive::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[strum(serialize_all = "lowercase")]
pub enum FftDirection {
    #[default]
    Forward = 0,
    Inverse = 1,
}

impl From<usize> for FftDirection {
    fn from(direction: usize) -> Self {
        match direction % 2 {
            0 => Self::Forward,
            _ => Self::Inverse,
        }
    }
}
impl From<FftDirection> for usize {
    fn from(direction: FftDirection) -> Self {
        direction as usize
    }
}
