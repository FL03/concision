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

    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U;
}
/// The [`ApplyOnce`] trait consumes the container and applies the given function to every
/// element before returning a new container with the results.
pub trait ApplyOnce<T> {
    type Cont<_T>;

    fn apply_once<F, U>(self, f: F) -> Self::Cont<U>
    where
        F: FnOnce(T) -> U;
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

impl<T> Apply<T> for &T
where
    T: Apply<T>,
{
    type Cont<V> = T::Cont<V>;

    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U,
    {
        Apply::apply(*self, f)
    }
}

impl<T> Apply<T> for &mut T
where
    T: Apply<T>,
{
    type Cont<V> = T::Cont<V>;

    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U,
    {
        Apply::apply(*self, f)
    }
}

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

impl<'a, A> Apply<&'a A> for &'a Option<A> {
    type Cont<V> = Option<V>;

    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&'a A) -> U,
    {
        self.as_ref().map(|a| f(a))
    }
}

impl<A, S, D> Apply<A> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(A) -> U,
    {
        self.mapv(f)
    }
}

impl<A, S, D> Apply<A> for &ArrayBase<S, D, A>
where
    A: ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Cont<V> = Array<V, D>;

    fn apply<F, U>(&self, f: F) -> Array<U, D>
    where
        F: Fn(A) -> U,
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

    fn apply<F, U>(&self, f: F) -> Array<U, D>
    where
        F: Fn(A) -> U,
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
