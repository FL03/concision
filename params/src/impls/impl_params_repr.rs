/*
    appellation: impl_params_repr <module>
    authors: @FL03
*/
use crate::params_base::ParamsBase;

use ndarray::{ArrayBase, DataOwned, Dimension, Ix1, Ix2, RawData};

impl<A, S, D> ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}

impl<A, S> ParamsBase<S, Ix1>
where
    S: RawData<Elem = A>,
{
    /// returns a new instance of the [`ParamsBase`] initialized using a _scalar_ bias along
    /// with the given, one-dimensional weight tensor.
    pub fn from_scalar_bias(bias: A, weights: ArrayBase<S, Ix1>) -> Self
    where
        A: Clone,
        S: DataOwned,
    {
        Self {
            bias: ArrayBase::from_elem((), bias),
            weights,
        }
    }
    /// returns the number of rows in the weights tensor
    pub fn nrows(&self) -> usize {
        self.weights().len()
    }
}

impl<A, S> ParamsBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    /// returns the number of columns in the weights tensor
    pub fn ncols(&self) -> usize {
        self.weights().ncols()
    }
    /// returns the number of rows in the weights tensor
    pub fn nrows(&self) -> usize {
        self.weights().nrows()
    }
}
