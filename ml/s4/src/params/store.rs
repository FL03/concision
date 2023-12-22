/*
    Appellation: store <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMParams;
use crate::core::prelude::GenerateRandom;
use crate::prelude::scanner;
use ndarray::prelude::{Array1, Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SSMStore<T = f64>
where
    T: Float,
{
    pub(crate) a: Array2<T>,
    pub(crate) b: Array2<T>,
    pub(crate) c: Array2<T>,
    pub(crate) d: Array2<T>,
}

impl<T> SSMStore<T>
where
    T: Float,
{
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

    pub fn ones(features: usize) -> Self {
        let a = Array2::<T>::ones((features, features));
        let b = Array2::<T>::ones((features, 1));
        let c = Array2::<T>::ones((1, features));
        let d = Array2::<T>::ones((1, 1));
        Self::new(a, b, c, d)
    }

    pub fn zeros(features: usize) -> Self {
        let a = Array2::<T>::zeros((features, features));
        let b = Array2::<T>::zeros((features, 1));
        let c = Array2::<T>::zeros((1, features));
        let d = Array2::<T>::zeros((1, 1));
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
    T: NdFloat,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Array2<T> {
        scanner(&self.a, &self.b, &self.c, u, x0)
    }
}

impl<T> SSMStore<T>
where
    T: Float + SampleUniform,
{
    pub fn uniform(features: usize) -> Self {
        let dk = T::one() / T::from(features).unwrap().sqrt();
        let a = Array2::<T>::uniform_between(dk, (features, features));
        let b = Array2::<T>::uniform_between(dk, (features, 1));
        let c = Array2::<T>::uniform_between(dk, (1, features));
        let d = Array2::<T>::ones((1, 1));
        Self::new(a, b, c, d)
    }
}

impl<T> ops::Index<SSMParams> for SSMStore<T>
where
    T: Float,
{
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

impl<T> ops::IndexMut<SSMParams> for SSMStore<T>
where
    T: Float,
{
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

impl<T> From<SSMStore<T>> for (Array2<T>, Array2<T>, Array2<T>, Array2<T>)
where
    T: Float,
{
    fn from(store: SSMStore<T>) -> Self {
        (store.a, store.b, store.c, store.d)
    }
}

impl<T> From<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)> for SSMStore<T>
where
    T: Float,
{
    fn from((a, b, c, d): (Array2<T>, Array2<T>, Array2<T>, Array2<T>)) -> Self {
        Self::new(a, b, c, d)
    }
}

impl<T> From<SSMStore<T>> for HashMap<SSMParams, Array2<T>>
where
    T: Float,
{
    fn from(store: SSMStore<T>) -> Self {
        let mut map = HashMap::new();

        map.insert(SSMParams::A, store.a);
        map.insert(SSMParams::B, store.b);
        map.insert(SSMParams::C, store.c);
        map.insert(SSMParams::D, store.d);
        map
    }
}

impl<T> FromIterator<(SSMParams, Array2<T>)> for SSMStore<T>
where
    T: Default + Float,
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

impl<T> IntoIterator for SSMStore<T>
where
    T: Float,
{
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
