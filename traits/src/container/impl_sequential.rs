/*
    Appellation: impl_sequential <module>
    Created At: 2025.12.10:21:44:00
    Contrib: @FL03
*/
use crate::container::SeqContainer;

macro_rules! impl_sequential {
    (@impl<$T:ident> $($name:ident)::*) => {
        impl<$T> SeqContainer for $($name)::*<$T> {
            fn len(&self) -> usize {
                self.len()
            }
        }
    };
     {$($($name:ident)::*<$T:ident>),* $(,)?} => {
        $(
            impl_sequential!(@impl<$T> $($name)::*);
        )*
    };
}

impl<C, T> SeqContainer for &C
where
    C: SeqContainer<Elem = T>,
    T: Sized,
{
    fn len(&self) -> usize {
        SeqContainer::len(*self)
    }
}

impl<C, T> SeqContainer for &mut C
where
    C: SeqContainer<Elem = T>,
    T: Sized,
{
    fn len(&self) -> usize {
        SeqContainer::len(*self)
    }
}

impl<T> SeqContainer for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> SeqContainer for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

#[cfg(feature = "alloc")]
impl_sequential! {
    alloc::vec::Vec<T>,
}
