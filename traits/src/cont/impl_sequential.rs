/*
    Appellation: impl_sequential <module>
    Created At: 2025.12.10:21:44:00
    Contrib: @FL03
*/
use super::Sequential;

macro_rules! impl_sequential {
    (@impl $($name:ident)::*<$T:ident>) => {
        impl<$T> $crate::cont::Sequential for $($name)::*<$T> {
            seal!();
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
    seal!();
}

impl<T> Sequential for &mut T
where
    T: Sequential,
{
    seal!();
}

impl<T> Sequential for [T] {
    seal!();
}

impl<T, const N: usize> Sequential for [T; N] {
    seal!();
}
#[cfg(feature = "alloc")]
impl_sequential! {
    alloc::vec::Vec<T>,
}
