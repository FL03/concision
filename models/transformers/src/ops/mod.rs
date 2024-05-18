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
    pub fn merge<A, S, D, E>(
        z: &mut ArrayBase<S, D>,
        swap: usize,
        with: usize,
    ) -> NdResult<CowArray<A, E>>
    where
        A: Clone,
        S: Data<Elem = A>,
        D: RemoveAxis<Smaller = E>,
        E: Dimension,
    {
        let cur = z.raw_dim().as_array_view().to_owned();
        let indicies = (0..cur.ndim()).filter(|&i| i != swap).collect::<Vec<_>>();
        let new_axis = cur[swap] * cur[with];
        let mut dim = cur.select(Axis(0), &indicies);
        dim[with - 1] = new_axis;

        // swap the head and sequence axes
        z.swap_axes(swap, with);
        // reshape the qkv matrix into a smaller dimension
        // z.to_shape((dim, Order::ColumnMajor))
        unimplemented!()
    }
    #[doc(hidden)]
    pub fn merge_simple<A, S, D, E>(
        z: &mut ArrayBase<S, D>,
        dim: E,
        swap: usize,
        with: usize,
    ) -> NdResult<CowArray<A, E>>
    where
        A: Clone,
        S: Data<Elem = A>,
        D: RemoveAxis<Smaller = E>,
        E: Dimension,
    {
        // swap the head and sequence axes
        z.swap_axes(swap, with);
        // reshape the qkv matrix into a smaller dimension
        z.to_shape((dim, Order::ColumnMajor))
    }

    pub fn merge_heads<A>(heads: &Array3<A>) -> NdResult<Array2<A>>
    where
        A: Clone,
    {
        let (n, seq, query) = heads.dim();
        let mut tmp = heads.clone();
        // swap the head and sequence axes
        tmp.swap_axes(0, 1);
        // reshape the qkv matrix into a 2d array
        tmp.into_shape((seq, n * query))
    }

    pub fn split_heads<T>(param: &Array2<T>, num_heads: usize) -> NdResult<Array3<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / num_heads;
        // reshape the qkv matrix into a 3d array
        let mut res = param
            .clone()
            .into_shape((param.shape()[0], num_heads, dim))?;
        // swap the sequence and head axes
        res.swap_axes(0, 1);
        Ok(res)
    }

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

    pub fn split_batch<T>(param: &Array3<T>, num_heads: usize) -> NdResult<Array4<T>>
    where
        T: Clone,
    {
        let dim = param.shape().last().unwrap() / num_heads;
        // reshape the qkv matrix into a 3d array
        let mut res =
            param
                .clone()
                .into_shape((param.shape()[0], param.shape()[1], num_heads, dim))?;
        // swap the sequence and head axes
        res.swap_axes(1, 2);
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn reshape_ops() {
        let dim_input: [usize; 3] = [2, 4, 6]; // (batch, seq, model)
        let dim_split = [2, 2, 4, 3]; // (batch, heads, seq, model)
        let data = Array::linspace(1., 48., 48).into_shape(dim_input).unwrap();

        let a = split_batch(&data, 2).unwrap();
        assert_eq!(a.shape(), &dim_split);
        assert_eq!(&a, &data.split(2).unwrap());
        let b = merge_batch(&a).unwrap();
        assert_eq!(b.shape(), &dim_input);
        assert_eq!(&b, &data);
    }
}
