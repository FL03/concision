/*
   Appellation: frames <module>
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

use std::iter::Enumerate;
use std::slice::Iter;

pub enum Indexers {
    Standard(Vec<usize>),
}

pub trait DataframeSpec<Data = usize> {
    fn aggregate(&self, source: &str) -> Self
    where
        Self: Sized;
    fn clean(&self, style: &str) -> Self
    where
        Self: Sized;
    fn create(&self) -> Self
    where
        Self: Sized;
    fn describe(&self) -> Self
    where
        Self: Sized;
}

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Dataframe<Dt> {
    pub index: Vec<usize>,
    pub data: Vec<Dt>,
}

impl<Dt: Clone> Dataframe<Dt> {
    pub fn constructor(index: Vec<usize>, data: Vec<Dt>) -> Self {
        Self { index, data }
    }

    pub fn new() -> Self {
        Self::constructor(Vec::new(), Vec::new())
    }

    pub fn from(data: Vec<Dt>) -> Self {
        Self::constructor(
            data.iter().enumerate().map(|pair| pair.0).collect(),
            data.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut df1: Dataframe<String> = Dataframe::new();
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }

    #[test]
    fn test_indexer() {
        let a: Vec<usize> = Vec::from([0, 1, 2, 3, 4, 5]);
        let index: Vec<usize> = a.iter().enumerate().map(|i| i.0).collect();
        assert_eq!(&a, &index)
    }
}
