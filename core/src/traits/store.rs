/*
    appellation: store <module>
    authors: @FL03
*/
/// The [`RawStore`] trait provides a generalized interface for all _containers_. The trait is
/// sealed, preventing any external implementations and is primarily used as the basis for
/// other traits, such as [`Sequential`].
pub trait RawStore {
    type Elem;

    private!();
}
/// The [`Sequential`] trait is a marker trait defining a sequential collection of elements.
/// It is sealed, preventing external implementations, and is used to indicate that a type can
/// be treated as a sequence of elements, such as arrays or vectors.
pub trait Sequential {
    private!();
}

macro_rules! impl_raw_store {
    (@impl $($name:ident)::*<$T:ident>) => {
        impl<$T> $crate::traits::store::RawStore for $($name)::*<$T> {
            type Elem = $T;

            seal!();
        }
    };
    {
        $(
            $($name:ident)::*<$T:ident>
        ),* $(,)?
    } => {
        $(
            impl_raw_store!(@impl $($name)::*<$T>);
        )*
    };
}

macro_rules! impl_sequential {
    (@impl $($name:ident)::*<$T:ident>) => {
        impl<$T> $crate::traits::store::Sequential for $($name)::*<$T> {
            seal!();
        }
    };
    {
        $(
            $($name:ident)::*<$T:ident>
        ),* $(,)?
    } => {
        $(
            impl_sequential!(@impl $($name)::*<$T>);
        )*
    };
}

impl<S, T> RawStore for &S
where
    S: RawStore<Elem = T>,
{
    type Elem = T;

    seal!();
}

impl<S, T> RawStore for &mut S
where
    S: RawStore<Elem = T>,
{
    type Elem = T;

    seal!();
}

impl<T> Sequential for &T
where
    T: Sequential,
{
    seal!();
}

impl<T> Sequential for &mut T
where
    T: Sequential,
{
    seal!();
}

impl<T> RawStore for [T] {
    type Elem = T;

    seal!();
}

impl<T, const N: usize> RawStore for [T; N] {
    type Elem = T;

    seal!();
}

impl<T> Sequential for [T] {
    seal!();
}

impl<T, const N: usize> Sequential for [T; N] {
    seal!();
}

impl_raw_store! {
    Option<T>
}

#[cfg(all(feature = "alloc", not(feature = "nightly")))]
mod impl_alloc {
    use super::RawStore;

    impl<K, V> RawStore for alloc::collections::BTreeMap<K, V> {
        type Elem = V;

        seal!();
    }

    impl_raw_store! {
        alloc::boxed::Box<T>,
        alloc::sync::Arc<T>,
        alloc::collections::BTreeSet<K>,
        alloc::vec::Vec<T>
    }

    impl_sequential! {
        alloc::vec::Vec<T>,
    }
}

#[cfg(all(feature = "alloc", feature = "nightly"))]
mod impl_alloc {
    use super::RawStore;
    use alloc::alloc::Allocator;
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::vec::Vec;

    impl<T, A> RawStore for alloc::boxed::Box<T, A>
    where
        A: Allocator + Clone,
    {
        type Elem = T;

        seal!();
    }

    impl<K, V, A> RawStore for BTreeMap<K, V, A>
    where
        A: Allocator + Clone,
    {
        type Elem = V;

        seal!();
    }

    impl<K, A> RawStore for BTreeSet<K, A>
    where
        A: Allocator + Clone,
    {
        type Elem = K;

        seal!();
    }

    impl<T, A> RawStore for Vec<T, A>
    where
        A: Allocator + Clone,
    {
        type Elem = T;

        seal!();
    }
}

#[cfg(feature = "std")]
mod impl_std {
    use super::RawStore;

    impl<K, V, S> RawStore for std::collections::HashMap<K, V, S> {
        type Elem = V;

        seal!();
    }

    impl<K, S> RawStore for std::collections::HashSet<K, S> {
        type Elem = K;

        seal!();
    }
}
