/*
    Appellation: wnb <module>
    Created At: 2025.11.28:21:21:42
    Contrib: @FL03
*/
use ndarray::iter as nditer;
use ndarray::{ArrayBase, Data, DataMut, Dimension, RawData};

/// A trait denoting an implementor with weights and associated methods
pub trait Weighted<S, D, A = <S as RawData>::Elem>: Sized
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Tensor<_S, _D, _A>
    where
        _D: Dimension,
        _S: RawData<Elem = _A>;
    /// returns the weights of the model
    fn weights(&self) -> &Self::Tensor<S, D, A>;
    /// returns a mutable reference to the weights of the model
    fn weights_mut(&mut self) -> &mut Self::Tensor<S, D, A>;
    /// replaces the current weights with the given weights
    fn replace_weights(&mut self, weights: Self::Tensor<S, D, A>) -> Self::Tensor<S, D, A> {
        core::mem::replace(self.weights_mut(), weights)
    }
    /// sets the weights of the model
    fn set_weights(&mut self, weights: Self::Tensor<S, D, A>) -> &mut Self {
        *self.weights_mut() = weights;
        self
    }
}

pub trait NdIter<A, D>
where
    D: Dimension,
{
    type Iter<'b, T>
    where
        T: 'b,
        Self: 'b;
    /// returns an iterator over the weights;
    fn nditer(&self) -> Self::Iter<'_, A>;
}

pub trait NdIterMut<A, D>
where
    D: Dimension,
{
    type IterMut<'b, T>
    where
        T: 'b,
        Self: 'b;
    /// returns a mutable iterator over the weights
    fn nditer_mut(&mut self) -> Self::IterMut<'_, A>;
}
pub trait Biased<S, D, A = <S as RawData>::Elem>: Weighted<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns the bias of the model
    fn bias(&self) -> &ArrayBase<S, D::Smaller, A>;
    /// returns a mutable reference to the bias of the model
    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A>;
    /// assigns the given bias to the current bias
    fn assign_bias(&mut self, bias: &ArrayBase<S, D::Smaller, A>) -> &mut Self
    where
        S: DataMut,
        S::Elem: Clone,
    {
        self.bias_mut().assign(bias);
        self
    }
    /// replaces the current bias with the given bias
    fn replace_bias(&mut self, bias: ArrayBase<S, D::Smaller, A>) -> ArrayBase<S, D::Smaller, A> {
        core::mem::replace(self.bias_mut(), bias)
    }
    /// sets the bias of the model
    fn set_bias(&mut self, bias: ArrayBase<S, D::Smaller, A>) -> &mut Self {
        *self.bias_mut() = bias;
        self
    }
    /// returns an iterator over the bias
    fn iter_bias<'a>(&'a self) -> nditer::Iter<'a, S::Elem, D::Smaller>
    where
        S: Data + 'a,
        D: 'a,
    {
        self.bias().iter()
    }
    /// returns a mutable iterator over the bias
    fn iter_bias_mut<'a>(&'a mut self) -> nditer::IterMut<'a, S::Elem, D::Smaller>
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

impl<A, S, D> NdIter<A, D> for ArrayBase<S, D, A>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    type Iter<'b, T>
        = nditer::Iter<'b, T, D>
    where
        T: 'b,
        Self: 'b;

    fn nditer(&self) -> Self::Iter<'_, A> {
        self.iter()
    }
}

impl<A, S, D> NdIterMut<A, D> for ArrayBase<S, D, A>
where
    S: DataMut<Elem = A>,
    D: Dimension,
{
    type IterMut<'b, T>
        = nditer::IterMut<'b, T, D>
    where
        T: 'b,
        Self: 'b;

    fn nditer_mut(&mut self) -> Self::IterMut<'_, A> {
        self.iter_mut()
    }
}
