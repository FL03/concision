/*
    Appellation: store <module>
    Created At: 2025.12.10:21:30:34
    Contrib: @FL03
*/

mod impl_data_container;
mod impl_raw_container;
mod impl_sequential;

/// The [`RawContainer`] trait provides a generalized interface for all _containers_. The trait is
/// sealed, preventing any external implementations and is primarily used as the basis for
/// other traits, such as [`Sequential`].
pub trait RawContainer {
    type Elem: ?Sized;
}
/// The [`SeqContainer`] trait is a marker trait defining a sequential collection of elements.
/// It is sealed, preventing external implementations, and is used to indicate that a type can
/// be treated as a sequence of elements, such as arrays or vectors.
pub trait SeqContainer: RawContainer {
    fn len(&self) -> usize;
}

/// The [`DataContainer`] trait works to extend the functionality of the [`RawContainer`]
pub trait DataContainer<T> {
    type Cont<_T>: ?Sized + RawContainer<Elem = _T>;
}

pub trait Container<T>: DataContainer<T> {
    fn apply<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: FnMut(&T) -> U;
}
