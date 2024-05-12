/*
    Appellation: ops <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
/// A trait for applying a function to a type
pub trait Apply<T, F> {
    type Output;

    fn apply<U>(&self, f: F) -> Self::Output
    where
        F: Fn(T) -> U;

    fn apply_mut<U>(&mut self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U;
}

pub trait ApplyOnce<T, F> {
    type Output;

    fn apply<U>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U;
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<T, F, S> ApplyOnce<T, F> for S
where
    S: Iterator<Item = T>,
{
    type Output = core::iter::Map<S, F>;

    fn apply<U>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        self.map(f)
    }
}
