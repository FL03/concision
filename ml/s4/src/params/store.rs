/*
    Appellation: store <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMParams;
use crate::core::prelude::GenerateRandom;
use ndarray::prelude::{Array, Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{vstack, Scalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;
use num::{Float, Num};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SSMStore<T = f64> {
    pub(crate) a: Array2<T>,
    pub(crate) b: Array2<T>,
    pub(crate) c: Array2<T>,
    pub(crate) d: Array2<T>,
}

impl<T> SSMStore<T>
where
    T: Clone + Num,
{
    pub fn ones(features: usize) -> Self {
        let a = Array2::ones((features, features));
        let b = Array2::ones((features, 1));
        let c = Array2::ones((1, features));
        let d = Array2::ones((1, 1));
        Self::new(a, b, c, d)
    }

    pub fn range(features: usize) -> Self
    where
        T: Float,
    {
        let a = Array::range(T::zero(), T::from(features * features).unwrap(), T::one())
            .into_shape((features, features))
            .unwrap();
        let b = Array::range(T::zero(), T::from(features).unwrap(), T::one()).insert_axis(Axis(1));
        let c = Array::range(T::zero(), T::from(features).unwrap(), T::one()).insert_axis(Axis(0));
        let d = Array2::zeros((1, 1));
        Self::new(a, b, c, d)
    }

    pub fn zeros(features: usize) -> Self {
        let a = Array2::zeros((features, features));
        let b = Array2::zeros((features, 1));
        let c = Array2::zeros((1, features));
        let d = Array2::zeros((1, 1));
        Self::new(a, b, c, d)
    }
}

impl<T> SSMStore<T> {
    pub fn new(a: Array2<T>, b: Array2<T>, c: Array2<T>, d: Array2<T>) -> Self {
        Self { a, b, c, d }
    }

    pub fn from_features(features: usize) -> Self
    where
        T: Default,
    {
        let a = Array2::<T>::default((features, features));
        let b = Array2::<T>::default((features, 1));
        let c = Array2::<T>::default((1, features));
        let d = Array2::<T>::default((1, 1));
        Self::new(a, b, c, d)
    }

    pub fn a(&self) -> &Array2<T> {
        &self.a
    }

    pub fn b(&self) -> &Array2<T> {
        &self.b
    }

    pub fn c(&self) -> &Array2<T> {
        &self.c
    }

    pub fn d(&self) -> &Array2<T> {
        &self.d
    }

    pub fn a_mut(&mut self) -> &mut Array2<T> {
        &mut self.a
    }

    pub fn b_mut(&mut self) -> &mut Array2<T> {
        &mut self.b
    }

    pub fn c_mut(&mut self) -> &mut Array2<T> {
        &mut self.c
    }

    pub fn d_mut(&mut self) -> &mut Array2<T> {
        &mut self.d
    }
}

impl<T> SSMStore<T>
where
    T: Scalar,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Result<Array2<T>, LinalgError> {
        let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
            *xs = self.a().dot(xs) + self.b().dot(&us);
            let y1 = self.c().dot(&xs.clone());
            Some(y1)
        };
        vstack(
            u.outer_iter()
                .scan(x0.clone(), step)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }
}

impl<T> SSMStore<T>
where
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, features: usize) -> Self {
        // let (lambda, p, b, _v) = dplr_hippo(features);
        self.c = Array2::<T>::random((1, features), StandardNormal);
        self.d = Array2::<T>::ones((1, 1));
        self
    }

    pub fn uniform(features: usize) -> Self {
        let dk = T::one() / T::from(features).unwrap().sqrt();
        let a = Array2::<T>::uniform_between(dk, (features, features));
        let b = Array2::<T>::uniform_between(dk, (features, 1));
        let c = Array2::<T>::uniform_between(dk, (1, features));
        let d = Array2::<T>::ones((1, 1));
        Self::new(a, b, c, d)
    }
}

impl<T> ops::Index<SSMParams> for SSMStore<T> {
    type Output = Array2<T>;

    fn index(&self, index: SSMParams) -> &Self::Output {
        use SSMParams::*;
        match index {
            A => self.a(),
            B => self.b(),
            C => self.c(),
            D => self.d(),
        }
    }
}

impl<T> ops::IndexMut<SSMParams> for SSMStore<T> {
    fn index_mut(&mut self, index: SSMParams) -> &mut Self::Output {
        use SSMParams::*;
        match index {
            A => self.a_mut(),
            B => self.b_mut(),
            C => self.c_mut(),
            D => self.d_mut(),
        }
    }
}

impl<T> From<SSMStore<T>> for (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    fn from(store: SSMStore<T>) -> Self {
        (store.a, store.b, store.c, store.d)
    }
}

impl<'a, T> From<&'a SSMStore<T>> for (&'a Array2<T>, &'a Array2<T>, &'a Array2<T>, &'a Array2<T>) {
    fn from(store: &'a SSMStore<T>) -> Self {
        (&store.a, &store.b, &store.c, &store.d)
    }
}

impl<T> From<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)> for SSMStore<T> {
    fn from((a, b, c, d): (Array2<T>, Array2<T>, Array2<T>, Array2<T>)) -> Self {
        Self::new(a, b, c, d)
    }
}

impl<T> From<SSMStore<T>> for HashMap<SSMParams, Array2<T>> {
    fn from(store: SSMStore<T>) -> Self {
        HashMap::from_iter(store.into_iter())
    }
}

impl<T> FromIterator<(SSMParams, Array2<T>)> for SSMStore<T>
where
    T: Clone + Default,
{
    fn from_iter<I: IntoIterator<Item = (SSMParams, Array2<T>)>>(iter: I) -> Self {
        let tmp = HashMap::<SSMParams, Array2<T>>::from_iter(iter);
        if tmp.is_empty() {
            Self::from_features(1)
        } else {
            let a = tmp
                .get(&SSMParams::A)
                .unwrap_or(&Array2::<T>::default((1, 1)))
                .clone();
            let b = tmp
                .get(&SSMParams::B)
                .unwrap_or(&Array2::<T>::default((1, 1)))
                .clone();
            let c = tmp
                .get(&SSMParams::C)
                .unwrap_or(&Array2::<T>::default((1, 1)))
                .clone();
            let d = tmp
                .get(&SSMParams::D)
                .unwrap_or(&Array2::<T>::default((1, 1)))
                .clone();
            Self::new(a, b, c, d)
        }
    }
}

impl<T> IntoIterator for SSMStore<T> {
    type Item = (SSMParams, Array2<T>);
    type IntoIter = std::collections::hash_map::IntoIter<SSMParams, Array2<T>>;

    fn into_iter(self) -> Self::IntoIter {
        HashMap::from(self).into_iter()
    }
}

// impl<'a, T> IntoIterator for &'a mut SSMStore<T> where T: Float {
//     type Item = (&'a SSMParams, &'a mut Array2<T>);
//     type IntoIter = std::collections::hash_map::IterMut<'a, SSMParams, Array2<T>>;

//     fn into_iter(self) -> Self::IntoIter {
//         HashMap::from(self).iter_mut()
//     }
// }
