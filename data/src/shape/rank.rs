/*
   Appellation: rank <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub enum Ranks<T> {
    Zero(T),
    One(Vec<T>),
    N(Vec<Self>),
}

pub struct Rank(pub usize);

impl Rank {
    pub fn new(rank: usize) -> Self {
        Self(rank)
    }

    pub fn rank(&self) -> usize {
        self.0
    }
}
