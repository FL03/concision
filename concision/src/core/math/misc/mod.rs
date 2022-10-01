/*
    Appellation: mod <module>
    Contributors: FL03 <jo3mccain@icloud.com> (https://gitlab.com/FL03)
    Description:
        ... Summary ...
*/
pub use self::factorials::*;

mod factorials;

use std::ops::{Add, Div, Mul, Sub};

pub enum Z {
    Signed,
    Unsigned,
}

pub enum Numbers {
    Integer,
    Float,
}

pub struct Point<T>(T);

pub trait BaseObject: Clone + Sized {}

pub trait Numerical<T>:
Add<Output=T> + Div<Output=T> + Mul<Output=T> + Sub<Output=T> + Sized
{}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrdPair<T: Numerical<T>>((Re<T>, Re<T>));

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Re<T: Numerical<T>>(T);

impl<T: Numerical<T>> Re<T> {
    pub fn new(data: T) -> Self {
        Self(data)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Im<T: Numerical<T>>(Re<T>);

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize, y: usize| x + y;
        let actual = f(4, 4);
        let expected: usize = 8;
        assert_eq!(actual, expected)
    }
}
