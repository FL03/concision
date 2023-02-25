/*
    Appellation: num <module>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/

pub mod complex;

use crate::Numerical;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct OrdPair<T: Numerical>((Re<T>, Re<T>));

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Re<T: Numerical>(T);

impl<T: Numerical> Re<T> {
    pub fn new(data: T) -> Self {
        Self(data)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Im<T: Numerical>(Re<T>);
