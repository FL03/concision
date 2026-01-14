/*
    appellation: impl_params_iter <module>
    authors: @FL03
*/
use crate::params_base::ParamsBase;

use crate::iter::{ParamsIter, ParamsIterMut};
use ndarray::iter as nditer;
use ndarray::{Axis, Data, DataMut, Dimension, RawData, RemoveAxis};

/// Here, we implement various iterators for the parameters and its constituents. The _core_
/// iterators are:
///
/// - immutable and mutable iterators over each parameter (weights and bias) respectively;
/// - an iterator over the parameters, which zips together an axis iterator over the columns of
///   the weights and an iterator over the bias;
impl<A, S, D> ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    /// an iterator of the parameters; the created iterator zips together an axis iterator over
    /// the columns of the weights and an iterator over the bias
    pub fn iter(&self) -> ParamsIter<'_, A, D>
    where
        D: RemoveAxis,
        S: Data,
    {
        ParamsIter {
            bias: self.bias().iter(),
            weights: self.weights().axis_iter(Axis(1)),
        }
    }
    /// returns a mutable iterator of the parameters, [`IterMut`], which essentially zips
    /// together a mutable axis iterator over the columns of the weights against a mutable
    /// iterator over the elements of the bias
    pub fn iter_mut(&mut self) -> ParamsIterMut<'_, A, D>
    where
        D: RemoveAxis,
        S: DataMut,
    {
        ParamsIterMut {
            bias: self.bias.iter_mut(),
            weights: self.weights.axis_iter_mut(Axis(0)),
        }
    }
    /// returns an iterator over the bias
    pub fn iter_bias(&self) -> nditer::Iter<'_, A, D::Smaller>
    where
        S: Data,
    {
        self.bias().iter()
    }
    /// returns a mutable iterator over the bias
    pub fn iter_bias_mut(&mut self) -> nditer::IterMut<'_, A, D::Smaller>
    where
        S: DataMut,
    {
        self.bias_mut().iter_mut()
    }
    /// returns an iterator over the weights
    pub fn iter_weights(&self) -> nditer::Iter<'_, A, D>
    where
        S: Data,
    {
        self.weights().iter()
    }
    /// returns a mutable iterator over the weights; see [`iter_mut`](ndarray::iter::IterMut) for more
    pub fn iter_weights_mut(&mut self) -> nditer::IterMut<'_, A, D>
    where
        S: DataMut,
    {
        self.weights_mut().iter_mut()
    }
    /// returns an iterator over the weights along the specified axis
    pub fn axis_iter_weights(&self, axis: Axis) -> nditer::AxisIter<'_, A, D::Smaller>
    where
        D: RemoveAxis,
        S: Data,
    {
        self.weights().axis_iter(axis)
    }
    /// returns a mutable iterator over the weights along the specified axis
    pub fn axis_iter_weights_mut(&mut self, axis: Axis) -> nditer::AxisIterMut<'_, A, D::Smaller>
    where
        D: RemoveAxis,
        S: DataMut,
    {
        self.weights_mut().axis_iter_mut(axis)
    }
}
