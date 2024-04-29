/*
   Appellation: plan <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct FftPlan {
    plan: Vec<usize>,
}

impl FftPlan {
    pub fn new(n: usize) -> Self {
        let mut plan = Vec::with_capacity(n);
        plan.extend(0..n);

        let mut rev = 0; // reverse
        let mut pos = 1; // position
        while pos < n {
            let mut bit = n >> 1;
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
        Self { plan }
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
        Self {
            plan: Vec::from_iter(iter),
        }
    }
}

impl IntoIterator for FftPlan {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut FftPlan {
    type Item = &'a mut usize;
    type IntoIter = std::slice::IterMut<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.plan.iter_mut()
    }
}
