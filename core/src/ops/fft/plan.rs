/*
   Appellation: plan <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[cfg(all(feature = "alloc", no_std))]
use alloc::vec::{self, Vec};
use core::slice;
#[cfg(feature = "std")]
use std::vec;

#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FftPlan {
    n: usize,
    plan: Vec<usize>,
}

impl FftPlan {
    pub fn new(n: usize) -> Self {
        let plan = Vec::with_capacity(n);
        Self { n, plan }
    }

    pub fn build(self) -> Self {
        let mut plan = Vec::with_capacity(self.n);
        plan.extend(0..self.n);

        let mut rev = 0; // reverse
        let mut pos = 1; // position
        while pos < self.n {
            let mut bit = self.n >> 1;
            while bit & rev != 0 {
                rev ^= bit;
                bit >>= 1;
            }
            rev ^= bit;
            // This is equivalent to adding 1 to a reversed number
            if pos < rev {
                // Only swap each element once
                plan.swap(pos, rev);
            }
            pos += 1;
        }
        Self { plan, ..self }
    }

    pub fn clear(&mut self) {
        self.n = 0;
        self.plan.clear();
    }

    pub fn plan(&self) -> &[usize] {
        &self.plan
    }
}

impl AsRef<[usize]> for FftPlan {
    fn as_ref(&self) -> &[usize] {
        &self.plan
    }
}

impl AsMut<[usize]> for FftPlan {
    fn as_mut(&mut self) -> &mut [usize] {
        &mut self.plan
    }
}

impl Extend<usize> for FftPlan {
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        self.plan.extend(iter);
    }
}

impl FromIterator<usize> for FftPlan {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let plan = Vec::from_iter(iter);
        Self {
            n: plan.len(),
            plan,
        }
    }
}

impl IntoIterator for FftPlan {
    type Item = usize;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut FftPlan {
    type Item = &'a mut usize;
    type IntoIter = slice::IterMut<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.iter_mut()
    }
}
