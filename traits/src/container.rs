/*
    Appellation: store <module>
    Created At: 2025.12.10:21:30:34
    Contrib: @FL03
*/

mod impl_container;
mod impl_raw_space;
mod impl_sequential;

/// The [`RawSpace`] trait provides a generalized interface for all _containers_. The trait is
/// sealed, preventing any external implementations and is primarily used as the basis for
/// other traits.
pub trait RawSpace {
    type Elem: ?Sized;
}
/// The [`SeqContainer`] trait is a marker trait defining a sequential collection of elements.
/// It is sealed, preventing external implementations, and is used to indicate that a type can
/// be treated as a sequence of elements, such as arrays or vectors.
pub trait SeqContainer: RawSpace {
    fn len(&self) -> usize;
}

/// A [`Container`] is a generalized abstraction for data structures that can hold elements of a
/// specific type. This trait is sealed to prevent external implementations and serves as a
/// foundational building block for more specialized container traits.
pub trait Container<T> {
    type Cont<_T>: ?Sized + RawSpace<Elem = _T>;
}

pub trait ContainerRef<T>: Container<T> {
    fn apply_ref<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&T) -> U;
}

pub trait ContainerOwned<T>: Container<T> {
    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&T) -> U;
}

pub trait ContainerMut<T>: Container<T> {
    fn apply_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T);
}
