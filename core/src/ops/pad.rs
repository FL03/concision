/*
   Appellation: pad <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::array_like;
use core::borrow::Cow;
use ndarray::prelude::{Array, ArrayBase, Dimension};
use ndarray::{AxisDescription, Data, Slice};
use num::{FromPrimitive, Num, Zero};
use strum::{AsRefStr, Display, EnumCount, EnumIs, EnumIter, VariantNames};

fn read_pad(nb_dim: usize, pad: &[[usize; 2]]) -> Cow<[[usize; 2]]> {
    if pad.len() == 1 && pad.len() < nb_dim {
        // The user provided a single padding for all dimensions
        Cow::from(vec![pad[0]; nb_dim])
    } else if pad.len() == nb_dim {
        Cow::from(pad)
    } else {
        panic!("Inconsistant number of dimensions and pad arrays");
    }
}

pub fn pad<S, A, D>(data: &ArrayBase<S, D>, pad: &[[usize; 2]], mode: PadMode<A>) -> Array<A, D>
where
    S: Data<Elem = A>,
    A: Copy + FromPrimitive + Num,
    D: Dimension,
{
    let pad = read_pad(data.ndim(), pad);
    let mut new_dim = data.raw_dim();
    for (ax, (&ax_len, pad)) in data.shape().iter().zip(pad.iter()).enumerate() {
        new_dim[ax] = ax_len + pad[0] + pad[1];
    }

    let mut padded = array_like(&data, new_dim, mode.init());
    pad_to(data, &pad, mode, &mut padded);
    padded
}

pub fn pad_to<S, A, D>(
    data: &ArrayBase<S, D>,
    pad: &[[usize; 2]],
    mode: PadMode<A>,
    output: &mut Array<A, D>,
) where
    S: Data<Elem = A>,
    A: Copy + FromPrimitive + Num,
    D: Dimension,
{
    let pad = read_pad(data.ndim(), pad);

    // Select portion of padded array that needs to be copied from the original array.
    output
        .slice_each_axis_mut(|ad| {
            let AxisDescription { axis, len, .. } = ad;
            let pad = pad[axis.index()];
            Slice::from(pad[0]..len - pad[1])
        })
        .assign(data);

    match mode.action() {
        PadAction::StopAfterCopy => { /* Nothing */ }
        _ => unimplemented!(),
    }
}
pub trait Pad<T> {
    fn pad(&self, pad: usize) -> Self;
}

// impl<T, D> Pad<T> for Array<T, D>
// where
//     T: Clone + Num,
//     D: Dimension,
// {
//     fn pad(&self, pad: usize) -> Self {
//         self.pad_with(pad, T::zero())
//     }

//     fn pad_with(&self, pad: usize, value: T) -> Self {
//         let mut pad = vec![value; pad];
//         pad.extend_from_slice(self);
//         pad.extend_from_slice(&vec![value; pad.len()]);
//         Array::from_vec(pad)
//     }
// }

#[derive(
    AsRefStr,
    Clone,
    Copy,
    Debug,
    Default,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    VariantNames,
)]
#[repr(u8)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize), serde(rename_all = "snake_case", untagged))]
#[strum(serialize_all = "snake_case")]
pub enum PadAction {
    Clipping,
    Lane,
    Reflecting,
    #[default]
    StopAfterCopy,
    Wrapping,
}

#[derive(
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
)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize), serde(rename_all = "lowercase", untagged))]
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

pub struct Padding<T> {
    pub mode: PadMode<T>,
    pub pad: usize,
}
