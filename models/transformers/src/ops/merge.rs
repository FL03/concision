/*
   Appellation: merge <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::NdResult;
use nd::prelude::*;
use nd::{Data, Order};

pub trait Merge {
    type Output;

    fn merge(self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<A, S> Merge for ArrayBase<S, Ix3>
where
    A: Clone,
    S: Data<Elem = A>,
{
    type Output = NdResult<Array2<A>>;

    fn merge(self) -> Self::Output {
        let (heads, seq, query) = self.dim();
        let mut tmp = self;
        // swap the head and sequence axes
        tmp.swap_axes(0, 1);
        // reshape the qkv matrix into a 2d array
        let res = tmp.to_shape(((seq, heads * query), Order::ColumnMajor))?;
        Ok(res.to_owned())
    }
}

impl<A, S> Merge for ArrayBase<S, Ix4>
where
    A: Clone,
    S: Data<Elem = A>,
{
    type Output = NdResult<Array3<A>>;

    fn merge(self) -> Self::Output {
        let (batch, heads, seq, query) = self.dim();
        let mut tmp = self;
        // swap the head and sequence axes
        tmp.swap_axes(1, 2);
        // reshape the qkv matrix into a 2d array
        let res = tmp.to_shape(((batch, seq, heads * query), Order::ColumnMajor))?;
        Ok(res.to_owned())
    }
}
