/*
    Appellation: sequential <module> [traits::misc]
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::traits::FromPrimitive;

/// A trait for sequential data structures;
/// This trait is implemented for iterators that have a known length.
pub trait Sequence<T> {
    const LENGTH: Option<usize> = None;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn elems(&self) -> T
    where
        T: FromPrimitive,
    {
        T::from_usize(self.len()).unwrap()
    }
}

pub trait SequenceIter {
    type Item;

    fn len(&self) -> usize;
}
/*
 ************* Implementations *************
*/
impl<T, I> SequenceIter for I
where
    I: ExactSizeIterator<Item = T>,
{
    type Item = T;

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Sequence<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Sequence<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> Sequence<T> for [T; N] {
    const LENGTH: Option<usize> = Some(N);

    fn len(&self) -> usize {
        N
    }
}
