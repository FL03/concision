use ndarray::{ArrayBase, Data, DataMut, Dimension, RawData};

pub trait Weighted<S, D>
where
    D: Dimension,
    S: RawData,
{
    /// returns the weights of the model
    fn weights(&self) -> &ArrayBase<S, D>;
    /// returns a mutable reference to the weights of the model
    fn weights_mut(&mut self) -> &mut ArrayBase<S, D>;
    /// assigns the given bias to the current weight
    fn assign_weights(&mut self, weights: &ArrayBase<S, D>) -> &mut Self
    where
        S: DataMut,
        S::Elem: Clone,
    {
        self.weights_mut().assign(weights);
        self
    }
    /// replaces the current weights with the given weights
    fn replace_weights(&mut self, weights: ArrayBase<S, D>) -> ArrayBase<S, D> {
        core::mem::replace(self.weights_mut(), weights)
    }
    /// sets the weights of the model
    fn set_weights(&mut self, weights: ArrayBase<S, D>) -> &mut Self {
        *self.weights_mut() = weights;
        self
    }
    /// returns an iterator over the weights
    fn iter_weights<'a>(&'a self) -> ndarray::iter::Iter<'a, S::Elem, D>
    where
        S: Data + 'a,
        D: 'a,
    {
        self.weights().iter()
    }
    /// returns a mutable iterator over the weights; see [`iter_mut`](ArrayBase::iter_mut) for more
    fn iter_weights_mut<'a>(&'a mut self) -> ndarray::iter::IterMut<'a, S::Elem, D>
    where
        S: DataMut + 'a,
        D: 'a,
    {
        self.weights_mut().iter_mut()
    }
}

pub trait Biased<S, D>: Weighted<S, D>
where
    D: Dimension,
    S: RawData,
{
    /// returns the bias of the model
    fn bias(&self) -> &ArrayBase<S, D::Smaller>;
    /// returns a mutable reference to the bias of the model
    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller>;
    /// assigns the given bias to the current bias
    fn assign_bias(&mut self, bias: &ArrayBase<S, D::Smaller>) -> &mut Self
    where
        S: DataMut,
        S::Elem: Clone,
    {
        self.bias_mut().assign(bias);
        self
    }
    /// replaces the current bias with the given bias
    fn replace_bias(&mut self, bias: ArrayBase<S, D::Smaller>) -> ArrayBase<S, D::Smaller> {
        core::mem::replace(self.bias_mut(), bias)
    }
    /// sets the bias of the model
    fn set_bias(&mut self, bias: ArrayBase<S, D::Smaller>) -> &mut Self {
        *self.bias_mut() = bias;
        self
    }
    /// returns an iterator over the bias
    fn iter_bias<'a>(&'a self) -> ndarray::iter::Iter<'a, S::Elem, D::Smaller>
    where
        S: Data + 'a,
        D: 'a,
    {
        self.bias().iter()
    }
    /// returns a mutable iterator over the bias
    fn iter_bias_mut<'a>(&'a mut self) -> ndarray::iter::IterMut<'a, S::Elem, D::Smaller>
    where
        S: DataMut + 'a,
        D: 'a,
    {
        self.bias_mut().iter_mut()
    }
}

/*
 ************* Implementations *************
*/
use crate::params::ParamsBase;

impl<S, D> Weighted<S, D> for ParamsBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    fn weights(&self) -> &ArrayBase<S, D> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.weights
    }
}

impl<S, D> Biased<S, D> for ParamsBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        &mut self.bias
    }
}
