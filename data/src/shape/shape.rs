/*
   Appellation: shape <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Rank;
use serde::{Deserialize, Serialize};
use std::ops::{self, Deref};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self(shape)
    }

    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn include(mut self, dim: usize) -> Self {
        self.0.push(dim);
        self
    }

    pub fn insert(&mut self, index: usize, dim: usize) {
        self.0.insert(index, dim)
    }

    /// Returns true if the strides are C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn push(&mut self, dim: usize) {
        self.0.push(dim)
    }

    pub fn rank(&self) -> Rank {
        self.0.len().into()
    }

    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn zeros(rank: usize) -> Self {
        Self(vec![0; rank])
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl AsMut<[usize]> for Shape {
    fn as_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Extend<usize> for Shape {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl From<usize> for Shape {
    fn from(dim: usize) -> Self {
        Self(vec![dim])
    }
}

impl From<(usize,)> for Shape {
    fn from(shape: (usize,)) -> Self {
        Self(vec![shape.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(shape: (usize, usize)) -> Self {
        Self(vec![shape.0, shape.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(shape: (usize, usize, usize)) -> Self {
        Self(vec![shape.0, shape.1, shape.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(shape: (usize, usize, usize, usize)) -> Self {
        Self(vec![shape.0, shape.1, shape.2, shape.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(shape: (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![shape.0, shape.1, shape.2, shape.3, shape.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(shape: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self(vec![shape.0, shape.1, shape.2, shape.3, shape.4, shape.5])
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut Shape {
    type Item = &'a mut usize;
    type IntoIter = std::slice::IterMut<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<Rank> for Shape {
    type Output = usize;

    fn index(&self, index: Rank) -> &Self::Output {
        &self.0[*index]
    }
}

impl ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl ops::IndexMut<Rank> for Shape {
    fn index_mut(&mut self, index: Rank) -> &mut Self::Output {
        &mut self.0[*index]
    }
}

impl ops::Index<ops::Range<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<ops::RangeTo<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<ops::RangeFrom<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<ops::RangeFull> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<ops::RangeInclusive<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::RangeInclusive<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl ops::Index<ops::RangeToInclusive<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: ops::RangeToInclusive<usize>) -> &Self::Output {
        &self.0[index]
    }
}

unsafe impl Send for Shape {}

unsafe impl Sync for Shape {}
