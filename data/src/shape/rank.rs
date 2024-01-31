/*
   Appellation: rank <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Rank
//!
//! The rank of a n-dimensional array describes the number of dimensions
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "lowercase")]
pub struct Rank(pub usize);

impl Rank {
    pub fn new(rank: usize) -> Self {
        Self(rank)
    }

    pub fn rank(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<usize> for Rank {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl AsMut<usize> for Rank {
    fn as_mut(&mut self) -> &mut usize {
        &mut self.0
    }
}

impl Deref for Rank {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Rank {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<usize> for Rank {
    fn from(rank: usize) -> Self {
        Self(rank)
    }
}

impl From<Rank> for usize {
    fn from(rank: Rank) -> Self {
        rank.0
    }
}

unsafe impl Send for Rank {}

unsafe impl Sync for Rank {}

// impl<T> TryFrom<T> for Rank
// where
//     T: NumCast,
// {
//     type Error = Box<dyn std::error::Error>;

//     fn try_from(value: T) -> Result<Self, Self::Error> {
//         if let Some(rank) = <usize as NumCast>::from(value) {
//             return Ok(Self(rank));
//         }
//         Err("Could not convert to Rank".into())
//     }
// }
