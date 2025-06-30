/*
    appellation: apply <module>
    authors: @FL03
*/

/// The [`Apply`] establishes an interface for _owned_ containers that are capable of applying
/// some function onto their elements.
pub trait Apply<T> {
    type Cont<_T>;

    fn apply<U, F>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U;
}
/// The [`ApplyMut`] trait mutates the each element of the container, in-place, using the given
/// function.
pub trait ApplyMut<T> {
    type Cont<_T>;

    fn apply_mut<'a, F>(&'a mut self, f: F)
    where
        T: 'a,
        F: FnMut(T) -> T;
}

/*
 ************* Implementations *************
*/
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use ndtensor::{Tensor, TensorBase};

impl<A, S, D> Apply<A> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply<V, F>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(A) -> V,
    {
        self.mapv(f)
    }
}

impl<A, S, D> Apply<A> for TensorBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Tensor<V, D>;

    fn apply<V, F>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(A) -> V,
    {
        self.map(f)
    }
}


impl<A, S, D> Apply<A> for &ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<A, S, D> Apply<A> for &mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply<B, F>(&self, f: F) -> Array<B, D>
    where
        F: Fn(A) -> B,
    {
        self.mapv(f)
    }
}

impl<A, S, D> ApplyMut<A> for ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply_mut<'a, F>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}

impl<A, S, D> ApplyMut<A> for &mut ArrayBase<S, D>
where
    A: ScalarOperand,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply_mut<'b, F>(&'b mut self, f: F)
    where
        A: 'b,
        F: FnMut(A) -> A,
    {
        self.mapv_inplace(f)
    }
}
