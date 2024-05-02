/*
   Appellation: activate <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod nl;

pub fn linear<T>(x: &T) -> T
where
    T: Clone,
{
    x.clone()
}

pub(crate) mod prelude {
    pub use super::nl::*;
}
