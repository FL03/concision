/*
   Appellation: like <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Array, ArrayBase, Axis, Data, Dimension};
use num::Num;

pub trait Like {
    type Item;

    fn like(&self) -> Self::Item;
}

pub trait Ones: Like
where
    Self::Item: num::One,
{
    fn ones_like(&self) -> Self;
}

pub trait ArrayLike {
    fn default_like(&self) -> Self;

    fn ones_like(&self) -> Self;

    fn zeros_like(&self) -> Self;
}


pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}



/*
    ******** implementations ********
*/

impl<T, D> ArrayLike for Array<T, D>
where
    T: Clone + Default + Num,
    D: Dimension,
{
    fn default_like(&self) -> Self
    where
        T: Default,
    {
        Array::default(self.dim())
    }

    fn ones_like(&self) -> Self {
        Array::ones(self.dim())
    }

    fn zeros_like(&self) -> Self {
        Array::zeros(self.dim())
    }
}

impl<A, S, D> Like for ArrayBase<S, D>
where
    A: Default,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Item = Array<A, D>;

    fn like(&self) -> Self::Item {
        Array::default(self.dim())
    }
}

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}