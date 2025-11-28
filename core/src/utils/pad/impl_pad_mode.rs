/*
    Appellation: impl_pad_mode <module>
    Created At: 2025.11.26:16:11:34
    Contrib: @FL03
*/
use super::{PadAction, PadMode};
use num_traits::Zero;

impl<T> From<T> for PadMode<T> {
    fn from(value: T) -> Self {
        PadMode::Constant(value)
    }
}

impl<T> PadMode<T> {
    pub fn as_pad_action(&self) -> PadAction {
        match *self {
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

    pub fn into_pad_action(self) -> PadAction {
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
