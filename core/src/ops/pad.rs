/*
   Appellation: pad <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{action::PadAction, mode::PadMode, utils::*};

pub(crate) mod action;
pub(crate) mod mode;

use nd::{Array, ArrayBase, DataOwned, Dimension};
use num::traits::{FromPrimitive, Num};

pub trait Pad<T> {
    type Output;

    fn pad(&self, mode: PadMode<T>, pad: &[[usize; 2]]) -> Self::Output;
}

impl<A, S, D> Pad<A> for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn pad(&self, mode: PadMode<A>, pad: &[[usize; 2]]) -> Self::Output {
        self::utils::pad(self, pad, mode)
    }
}

pub struct Padding<T> {
    pub(crate) action: PadAction,
    pub(crate) mode: PadMode<T>,
    pub(crate) pad: Vec<[usize; 2]>,
    pub(crate) padding: usize,
}

impl<T> Padding<T> {
    pub fn new() -> Self {
        Self {
            action: PadAction::default(),
            mode: PadMode::default(),
            pad: Vec::new(),
            padding: 0,
        }
    }

    pub fn pad(&self) -> &[[usize; 2]] {
        &self.pad
    }

    pub fn with_action(mut self, action: PadAction) -> Self {
        self.action = action;
        self
    }

    pub fn with_mode(mut self, mode: PadMode<T>) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}

mod utils {
    use super::{PadAction, PadMode};
    use crate::traits::ArrayLike;
    use nd::{Array, ArrayBase, AxisDescription, Data, DataOwned, Dimension, Slice};
    use num::{FromPrimitive, Num};

    #[cfg(all(feature = "alloc", no_std))]
    use alloc::borrow::Cow;
    #[cfg(feature = "std")]
    use std::borrow::Cow;

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

    pub fn pad<A, S, D>(data: &ArrayBase<S, D>, pad: &[[usize; 2]], mode: PadMode<A>) -> Array<A, D>
    where
        A: Copy + FromPrimitive + Num,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        let pad = read_pad(data.ndim(), pad);
        let mut new_dim = data.raw_dim();
        for (ax, (&ax_len, pad)) in data.shape().iter().zip(pad.iter()).enumerate() {
            new_dim[ax] = ax_len + pad[0] + pad[1];
        }

        // let mut padded = array_like(&data, new_dim, mode.init());
        let mut padded = data.array_like(new_dim, mode.init()).to_owned();
        pad_to(data, &pad, mode, &mut padded);
        padded
    }

    pub fn pad_to<A, S, D>(
        data: &ArrayBase<S, D>,
        pad: &[[usize; 2]],
        mode: PadMode<A>,
        output: &mut Array<A, D>,
    ) where
        A: Copy + FromPrimitive + Num,
        D: Dimension,
        S: Data<Elem = A>,
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
}
