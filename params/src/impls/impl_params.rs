/*
    appellation: impl_params <module>
    authors: @FL03
*/
use crate::params_base::ParamsBase;
use crate::traits::{Biased, Weighted};
use core::iter::Once;
use ndarray::{ArrayBase, Data, DataOwned, Dimension, Ix1, Ix2, RawData};

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
    /// returns the number of rows in the weights matrix
    pub fn nrows(&self) -> usize {
        self.weights().len()
    }
}

impl<A, S> ParamsBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    /// returns the number of columns in the weights matrix
    pub fn ncols(&self) -> usize {
        self.weights().ncols()
    }
    /// returns the number of rows in the weights matrix
    pub fn nrows(&self) -> usize {
        self.weights().nrows()
    }
}

impl<A, S, D> Weighted<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn weights(&self) -> &ArrayBase<S, D, A> {
        self.weights()
    }

    fn weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.weights_mut()
    }
}

impl<A, S, D> Biased<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn bias(&self) -> &ArrayBase<S, D::Smaller, A> {
        self.bias()
    }

    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A> {
        self.bias_mut()
    }
}

impl<A, S, D> core::fmt::Debug for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("bias", self.bias())
            .field("weights", self.weights())
            .finish()
    }
}

impl<A, S, D> core::fmt::Display for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "{{ bias: {}, weights: {} }}",
            self.bias(),
            self.weights()
        )
    }
}

impl<A, S, D> Clone for ParamsBase<S, D, A>
where
    D: Dimension,
    S: ndarray::RawDataClone<Elem = A>,
    A: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.bias().clone(), self.weights().clone())
    }
}

impl<A, S, D> Copy for ParamsBase<S, D, A>
where
    D: Dimension + Copy,
    <D as Dimension>::Smaller: Copy,
    S: ndarray::RawDataClone<Elem = A> + Copy,
    A: Copy,
{
}

impl<A, S, D> PartialEq for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> PartialEq<&ParamsBase<S, D, A>> for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &&ParamsBase<S, D, A>) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> PartialEq<&mut ParamsBase<S, D, A>> for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &&mut ParamsBase<S, D, A>) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> Eq for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: Eq,
{
}

impl<A, S, D> IntoIterator for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Item = ParamsBase<S, D, A>;
    type IntoIter = Once<ParamsBase<S, D, A>>;

    fn into_iter(self) -> Self::IntoIter {
        core::iter::once(self)
    }
}
