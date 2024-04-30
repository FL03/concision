/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{merge::*, split::*, utils::*};

pub(crate) mod merge;
pub(crate) mod split;

pub(crate) mod utils {
    use ndarray::prelude::{Array2, Array3, Array4};
    use ndarray::ShapeError;
    // use num::Float;

    pub fn merge_heads<T>(heads: &Array3<T>) -> Result<Array2<T>, ShapeError>
    where
        T: Clone,
    {
        let (n, seq, query) = heads.dim();
        let mut tmp = heads.clone();
        // swap the head and sequence axes
        tmp.swap_axes(0, 1);
        // reshape the qkv matrix into a 2d array
        tmp.into_shape((seq, n * query))
    }

    pub fn split_heads<T>(param: &Array2<T>, num_heads: usize) -> Result<Array3<T>, ShapeError>
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

    pub fn merge_batch<T>(heads: &Array4<T>) -> Result<Array3<T>, ShapeError>
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

    pub fn split_batch<T>(param: &Array3<T>, num_heads: usize) -> Result<Array4<T>, ShapeError>
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
