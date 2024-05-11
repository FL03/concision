/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::pad::PadAction;
use num::Zero;
use smart_default::SmartDefault;
use strum::{AsRefStr, Display, EnumCount, EnumIs, VariantNames};

#[derive(
    AsRefStr,
    Clone,
    Copy,
    Debug,
    Display,
    EnumCount,
    EnumIs,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    SmartDefault,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[repr(C)]
#[strum(serialize_all = "snake_case")]
pub enum PadMode<T> {
    Constant(T),
    Edge,
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Reflect,
    Symmetric,
    #[default]
    Wrap,
}

impl<T> From<T> for PadMode<T> {
    fn from(value: T) -> Self {
        PadMode::Constant(value)
    }
}

impl<T> PadMode<T> {
    pub(crate) fn action(&self) -> PadAction {
        match self {
            PadMode::Constant(_) => PadAction::StopAfterCopy,
            PadMode::Edge => PadAction::Clipping,
            PadMode::Maximum => PadAction::Clipping,
            PadMode::Mean => PadAction::Clipping,
            PadMode::Median => PadAction::Clipping,
            PadMode::Minimum => PadAction::Clipping,
            PadMode::Mode => PadAction::Clipping,
            PadMode::Reflect => PadAction::Reflecting,
            PadMode::Symmetric => PadAction::Reflecting,
            PadMode::Wrap => PadAction::Wrapping,
        }
    }
    pub fn init(&self) -> T
    where
        T: Copy + Zero,
    {
        match *self {
            PadMode::Constant(v) => v,
            _ => T::zero(),
        }
    }
}
