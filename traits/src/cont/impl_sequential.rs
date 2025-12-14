/*
    Appellation: impl_sequential <module>
    Created At: 2025.12.10:21:44:00
    Contrib: @FL03
*/
use super::Sequential;

macro_rules! impl_sequential {
    (@impl $($name:ident)::*<$T:ident>) => {
        impl<$T> $crate::cont::Sequential for $($name)::*<$T> {
            fn len(&self) -> usize {
                self.len()
            }
        }
    };
     {$($($name:ident)::*<$T:ident>),* $(,)?} => {
        $(
            impl_sequential!(@impl $($name)::*<$T>);
        )*
    };
}

impl<T> Sequential for &T
where
    T: Sequential,
{
    fn len(&self) -> usize {
        Sequential::len(*self)
    }
}

impl<T> Sequential for &mut T
where
    T: Sequential,
{
    fn len(&self) -> usize {
        Sequential::len(*self)
    }
}

impl<T> Sequential for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> Sequential for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

#[cfg(feature = "alloc")]
impl_sequential! {
    alloc::vec::Vec<T>,
}
