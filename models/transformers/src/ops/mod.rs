/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{merge::*, split::*, utils::*};

pub(crate) mod merge;
pub(crate) mod split;

pub(crate) mod utils {
    use concision::NdResult;
    use nd::prelude::*;
    use nd::{Data, Order, RemoveAxis};

    pub(crate) fn order(row_major: bool) -> Order {
        if row_major {
            Order::RowMajor
        } else {
            Order::ColumnMajor
        }
    }

    #[deprecated(
        since = "0.1.14",
        note = "Please use the `Merge::merge` method instead"
    )]
    pub fn merge<A, S, D>(
        arr: &ArrayBase<S, D>,
        src: usize,
        tgt: usize,
    ) -> NdResult<Array<A, D::Smaller>>
    where
        A: Clone,
        D: RemoveAxis,
        S: Data<Elem = A>,
        D::Smaller: Dimension,
        ArrayBase<S, D>: Clone,
    {
        _merge(arr, src, tgt, false)
    }
    #[deprecated(
        since = "0.1.14",
        note = "Please use the `Split::Split` method instead"
    )]
    pub fn split<A, S, D, E>(arr: &ArrayBase<S, D>, h: usize) -> NdResult<Array<A, E>>
    where
        A: Clone,
        D: Dimension<Larger = E>,
        E: RemoveAxis<Smaller = D>,
        S: Data<Elem = A>,
        ArrayBase<S, D>: Clone,
    {
        _split(arr, h, true)
    }

    pub(crate) fn _merge<A, S, D>(
        arr: &ArrayBase<S, D>,
        src: usize,
        tgt: usize,
        row_major: bool,
    ) -> NdResult<Array<A, D::Smaller>>
    where
        A: Clone,
        D: RemoveAxis,
        S: Data<Elem = A>,
        D::Smaller: Dimension,
        ArrayBase<S, D>: Clone,
    {
        let shape = _merge_dim(&arr.raw_dim(), src);
        let mut head = arr.clone();
        head.swap_axes(src, tgt);
        head.to_shape((shape, order(row_major)))
            .map(|x| x.to_owned())
    }

    pub(crate) fn _split<A, S, D, E>(
        arr: &ArrayBase<S, D>,
        h: usize,
        row_major: bool,
    ) -> NdResult<Array<A, E>>
    where
        A: Clone,
        D: Dimension<Larger = E>,
        E: RemoveAxis<Smaller = D>,
        S: Data<Elem = A>,
        ArrayBase<S, D>: Clone,
    {
        let src = if arr.ndim() >= 2 { arr.ndim() - 2 } else { 0 };
        let tgt = src + 1;
        let shape: E = _split_dim(&arr.raw_dim(), h);
        let mut head = arr.clone();

        head.swap_axes(src, tgt);
        head.to_shape((shape, order(row_major)))
            .map(|x| x.to_owned())
    }
    /// Creates the new dimension after merging two axes.
    pub(crate) fn _merge_dim<D>(dim: &D, src: usize) -> D::Smaller
    where
        D: RemoveAxis,
        D::Smaller: Dimension,
    {
        // create a new dimension with one less axis; initialized with zeros
        let mut new_dim = <D as Dimension>::Smaller::zeros(dim.ndim() - 1);
        // create a mutable vector from the slice
        let mut shape = dim.slice().to_vec();
        // multiply the last axis by the target
        shape[new_dim.ndim()] *= shape[src];
        // remove the last dimension
        shape.remove(src);

        new_dim.slice_mut().copy_from_slice(&shape);
        new_dim
    }

    pub(crate) fn _split_dim<D>(dim: &D::Smaller, h: usize) -> D
    where
        D: RemoveAxis,
        D::Smaller: Dimension,
    {
        let rank = dim.ndim() + 1;
        // create a new dimension with one less axis; initialized with zeros
        let mut new_dim = D::zeros(rank);
        // create a mutable vector from the slice
        let mut shape = dim.slice().to_vec();
        // get and remove the last axis
        let bx = shape.pop().unwrap() / h;
        // extend the shape with the new axes
        shape.push(h);
        shape.push(bx);
        shape.swap(rank - 2, rank - 3);
        // copy the values into the new dimension
        new_dim.slice_mut().copy_from_slice(&shape);
        new_dim
    }

    pub fn split_heads<T>(param: &Array2<T>, h: usize) -> NdResult<Array3<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / h;
        // reshape the qkv matrix into a 3d array
        let mut res = param.clone().into_shape((param.shape()[0], h, dim))?;
        // swap the sequence and head axes
        res.swap_axes(0, 1);
        Ok(res)
    }

    pub fn split_batch<T>(param: &Array3<T>, h: usize) -> NdResult<Array4<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / h;
        // reshape the qkv matrix into a 3d array
        let mut res = param
            .clone()
            .into_shape((param.shape()[0], param.shape()[1], h, dim))?;
        // swap the sequence and head axes
        res.swap_axes(1, 2);
        Ok(res)
    }
}
