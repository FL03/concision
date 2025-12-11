/*
    Appellation: hkt <module>
    Contrib: @FL03
*/

/// The [`HKT`] trait defines an interface for higher-kinded types (HKT).
pub trait HKT<T> {
    type Cont<_T>;
}

/// The [`Functor`] trait extends the [`HKT`] trait to provide a way to map over its content(s)
/// using a function `f` that transforms values of type `T` into values of type `U`.
pub trait Functor<T>: HKT<T> {
    fn mapf<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: Fn(&T) -> U;
}

pub trait Applicative<T>: Functor<T> {
    fn pure(value: T) -> Self::Cont<T>;
}

/*
 *************  Implementations  *************
*/
macro_rules! hkt {
    (@impl $($t:ident)::*<$T:ident>) => {
        impl<$T> HKT<$T> for $($t)::*<$T> {
            type Cont<_T> = $($t)::*<_T>;
        }
    };
    (
        $(
            $($t:ident)::*<$T:ident>
        ),* $(,)?
    ) => {
        $(
            hkt!(@impl $($t)::*<$T>);
        )*
    };
}

hkt! {
    core::option::Option<T>
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

impl<T> Functor<T> for Option<T> {
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
        assert_eq!(opt.mapf(|&x| x as f32 + 1.25), Some(43.25));
    }
}
