/*
    Appellation: hkt <module>
    Contrib: @FL03
*/

/// The [`HKT`] trait defines an interface for higher-kinded types (HKT).
pub trait HKT<T> {
    type Cont<_T>: ?Sized;
}

/// The [`Functor`] trait extends the [`HKT`] trait to provide a way to map over its content(s)
/// using a function `f` that transforms values of type `T` into values of type `U`.
pub trait Functor<T>: HKT<T> {
    fn mapf<F, U>(self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U;
}
/// An alternative version of the [`Functor`] trait that works with references to the content
pub trait FunctorRef<T>: HKT<T> {
    fn mapf<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&T) -> U;
}

// pub trait Applicative<T>: Functor<T> {
//     fn pure(value: T) -> Self::Cont<T>;
// }

// pub trait Monad<T>: Applicative<T> {
//     fn flat_map<F, U>(&self, f: F) -> Self::Cont<U>
//     where
//         F: Fn(&mut T, &T) -> Self::Cont<U>;
// }

/*
 *************  Implementations  *************
*/
macro_rules! hkt {
    (@impl $($cont:ident)::*<$T:ident>) => {
        impl<$T> HKT<$T> for $($cont)::*<$T> {
            type Cont<_T> = $($cont)::*<_T>;
        }
    };
    ($($($cont:ident)::*<$T:ident>),* $(,)?) => {
        $(hkt!(@impl $($cont)::*<$T>);)*
    };
}

impl<C, T> HKT<T> for &C
where
    C: HKT<T>,
{
    type Cont<U> = C::Cont<U>;
}

impl<C, T> HKT<T> for &mut C
where
    C: HKT<T>,
{
    type Cont<U> = C::Cont<U>;
}

impl<T> HKT<T> for [T] {
    type Cont<U> = [U];
}

impl<const N: usize, T> HKT<T> for [T; N] {
    type Cont<U> = [U; N];
}

impl<T, E> HKT<T> for core::result::Result<T, E> {
    type Cont<U> = core::result::Result<U, E>;
}

hkt! {
    core::option::Option<T>,
}

#[cfg(feature = "alloc")]
hkt! {
    alloc::vec::Vec<T>,
    alloc::boxed::Box<T>,
    alloc::rc::Rc<T>,
    alloc::rc::Weak<T>,
    alloc::sync::Arc<T>,
    alloc::collections::BinaryHeap<T>,
    alloc::collections::BTreeSet<T>,
    alloc::collections::LinkedList<T>,
    alloc::collections::VecDeque<T>,
}

#[cfg(feature = "std")]
hkt! {
    std::cell::Cell<T>,
    std::cell::OnceCell<T>,
    std::cell::RefCell<T>,
    std::sync::Mutex<T>,
    std::sync::RwLock<T>,
    std::sync::LazyLock<T>,
    std::collections::HashSet<V>,
}

#[cfg(feature = "alloc")]
impl<K, V> HKT<V> for alloc::collections::BTreeMap<K, V> {
    type Cont<U> = alloc::collections::BTreeMap<K, U>;
}

#[cfg(feature = "std")]
impl<K, V> HKT<V> for std::collections::HashMap<K, V> {
    type Cont<U> = std::collections::HashMap<K, U>;
}

#[cfg(feature = "hashbrown")]
impl<K, V, S> HKT<V> for hashbrown::HashMap<K, V, S> {
    type Cont<U> = hashbrown::HashMap<K, U, S>;
}

#[cfg(feature = "hashbrown")]
impl<T, S> HKT<T> for hashbrown::HashSet<T, S> {
    type Cont<U> = hashbrown::HashSet<U, S>;
}

impl<A, S, D> HKT<A> for ndarray::ArrayBase<S, D, A>
where
    S: ndarray::RawData<Elem = A>,
    D: ndarray::Dimension,
{
    type Cont<U> = ndarray::ArrayRef<U, D>;
}

impl<T> Functor<T> for Option<T> {
    fn mapf<F, U>(self, f: F) -> Self::Cont<U>
    where
        F: Fn(T) -> U,
    {
        self.map(f)
    }
}

impl<T> FunctorRef<T> for Option<T> {
    fn mapf<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&T) -> U,
    {
        self.as_ref().map(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option() {
        let opt = Some(42u8);
        assert_eq!(opt.mapf(|x| x as f32 + 1.25), Some(43.25));
    }
}
