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

    #[doc(hidden)]
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
        merger(arr, src, tgt, Order::RowMajor)
    }

    pub(crate) fn merger<A, S, D>(
        arr: &ArrayBase<S, D>,
        src: usize,
        tgt: usize,
        order: Order,
    ) -> NdResult<Array<A, D::Smaller>>
    where
        A: Clone,
        D: RemoveAxis,
        S: Data<Elem = A>,
        D::Smaller: Dimension,
        ArrayBase<S, D>: Clone,
    {
        let shape = merge_dims(arr.raw_dim(), src);
        let mut head = arr.clone();
        head.swap_axes(src, tgt);
        head.to_shape((shape, order)).map(|x| x.to_owned())
    }

    #[doc(hidden)]
    pub fn merge_dims<D>(dim: D, src: usize) -> D::Smaller
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

    #[doc(hidden)]
    pub fn merge_batch<T>(heads: &Array4<T>) -> NdResult<Array3<T>>
    where
        T: Clone,
    {
        let (batch, n, seq, query) = heads.dim();
        let mut tmp = heads.clone();
        // swap the head and sequence axes
        tmp.swap_axes(1, 2);
        // reshape the qkv matrix into a 2d array
        tmp.into_shape((batch, seq, n * query))
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
