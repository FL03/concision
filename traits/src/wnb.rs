use ndarray::{ArrayBase, Data, DataMut, Dimension, RawData};

pub trait Weighted<S, D, A = <S as RawData>::Elem>: Sized
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns the weights of the model
    fn weights(&self) -> &ArrayBase<S, D, A>;
    /// returns a mutable reference to the weights of the model
    fn weights_mut(&mut self) -> &mut ArrayBase<S, D, A>;
    /// assigns the given bias to the current weight
    fn assign_weights(&mut self, weights: &ArrayBase<S, D, A>) -> &mut Self
    where
        S: DataMut,
        S::Elem: Clone,
    {
        self.weights_mut().assign(weights);
        self
    }
    /// replaces the current weights with the given weights
    fn replace_weights(&mut self, weights: ArrayBase<S, D, A>) -> ArrayBase<S, D, A> {
        core::mem::replace(self.weights_mut(), weights)
    }
    /// sets the weights of the model
    fn set_weights(&mut self, weights: ArrayBase<S, D, A>) -> &mut Self {
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
