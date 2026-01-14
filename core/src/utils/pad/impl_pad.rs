/*
    Appellation: impl_pad <module>
    Created At: 2025.11.26:16:12:28
    Contrib: @FL03
*/
use super::{Pad, PadAction, PadMode};
use concision_traits::ArrayLike;
use ndarray::{Array, ArrayBase, AxisDescription, DataOwned, Dimension, Slice};
use num_traits::{FromPrimitive, Num};

fn reader(ndim: usize, pad: &[[usize; 2]]) -> Option<Vec<[usize; 2]>> {
    debug_assert!(pad.len() == ndim, "Inconsistent dimensions for padding");
    if pad.len() != ndim {
        return None;
    }
    Some(pad.to_vec())
}

fn apply_padding<A, S, D>(
    data: &ArrayBase<S, D, A>,
    pad: &[[usize; 2]],
    mode: PadMode<A>,
    output: &mut Array<A, D>,
) -> Option<bool>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    let pad = reader(data.ndim(), pad)?;

    // Select portion of padded array that needs to be copied from the original array.
    output
        .slice_each_axis_mut(|ad| {
            let AxisDescription { axis, len, .. } = ad;
            let pad = pad[axis.index()];
            Slice::from(pad[0]..len - pad[1])
        })
        .assign(data);

    match mode.into_pad_action() {
        PadAction::StopAfterCopy => {
            // Do nothing
            Some(true)
        }
        _ => unimplemented!(),
    }
}

pub fn pad<A, S, D>(
    data: &ArrayBase<S, D, A>,
    padding: &[[usize; 2]],
    mode: PadMode<A>,
) -> Array<A, D>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    let pad = reader(data.ndim(), padding).expect("Inconsistent dimensions for padding");
    let mut dim = data.raw_dim();
    for (ax, (&ax_len, pad)) in data.shape().iter().zip(pad.iter()).enumerate() {
        dim[ax] = ax_len + pad[0] + pad[1];
    }

    let mut padded = data.array_like(dim, mode.init()).to_owned();
    apply_padding(data, &pad, mode, &mut padded).expect("Failed to apply padding");
    padded
}

impl<A, S, D> Pad<A> for ArrayBase<S, D, A>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn pad(&self, mode: PadMode<A>, padding: &[[usize; 2]]) -> Self::Output {
        pad(self, padding, mode)
    }
}
