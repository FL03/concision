/*
    appellation: apply <module>
    authors: @FL03
*/
/// The [`CallInto`] trait is a consuming interface for passing an object into a single-valued
/// function. While the intended affect is the same as [`CallOn`], the difference is that
/// [`CallInto`] enables a transfer of ownership instead of relyin upon a reference.
pub trait CallInto<T> {
    type Output;

    /// The `call_into` method allows an object to be passed into a function that takes ownership
    /// of the object. This is useful for cases where you want to perform an operation on an
    /// object and consume it in the process.
    fn call_into<F>(self, f: F) -> Self::Output
    where
        F: FnOnce(T) -> Self::Output;
}
/// The [`CallOn`] trait enables an object to be passed onto a unary, or single value, function
/// that is applied to the object.
pub trait CallOn<T>: CallInto<T> {
    /// The `call_on` method allows an object to be passed onto a function that takes a reference
    /// to the object. This is useful for cases where you want to perform an operation on
    /// an object without needing to extract it from a container or context.
    fn call_on<F>(&self, f: F) -> Self::Output
    where
        F: FnMut(&T) -> Self::Output;
}
/// The [`CallOnMut`] is a supertrait of the [`CallInto`] trait that enables an object to be
/// passed onto a unary, or single value, function that is applied to the object, but with the
/// ability to mutate the object in-place.
pub trait CallInPlace<T>: CallInto<T> {
    /// The `call_on_mut` method allows an object to be passed onto a function that takes a mutable reference
    /// to the object. This is useful for cases where you want to perform an operation on
    /// an object and mutate it in the process.
    fn call_inplace<F>(&mut self, f: F) -> Self::Output
    where
        F: FnMut(&mut T) -> Self::Output;
}

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

impl<T> CallInto<T> for T {
    type Output = T;

    fn call_into<F>(self, f: F) -> Self::Output
    where
        F: FnOnce(T) -> Self::Output,
    {
        f(self)
    }
}

impl<T> CallOn<T> for T
where
    T: CallInto<T>,
{
    fn call_on<F>(&self, mut f: F) -> Self::Output
    where
        F: FnMut(&T) -> Self::Output,
    {
        f(self)
    }
}

impl<T> CallInPlace<T> for T
where
    T: CallInto<T>,
{
    fn call_inplace<F>(&mut self, mut f: F) -> Self::Output
    where
        F: FnMut(&mut T) -> Self::Output,
    {
        f(self)
    }
}

impl<A, S, D> Apply<A> for ArrayBase<S, D, A>
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

// impl<A, S, D> Apply<A> for TensorBase<S, D>
// where
//     A: ScalarOperand,
//     D: Dimension,
//     S: Data<Elem = A>,
// {
//     type Cont<V> = Tensor<V, D>;

//     fn apply<V, F>(&self, f: F) -> Self::Cont<V>
//     where
//         F: Fn(A) -> V,
//     {
//         self.map(f)
//     }
// }

impl<A, S, D> Apply<A> for &ArrayBase<S, D, A>
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

impl<A, S, D> Apply<A> for &mut ArrayBase<S, D, A>
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

impl<A, S, D> ApplyMut<A> for ArrayBase<S, D, A>
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

impl<A, S, D> ApplyMut<A> for &mut ArrayBase<S, D, A>
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
