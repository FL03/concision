/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::Inverse;

use ndarray::prelude::{Array2, NdFloat};

use nshare::{ToNalgebra, ToNdarray2};
use num::{Float, ToPrimitive};

pub fn discretize<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
) -> anyhow::Result<(Array2<T>, Array2<T>, Array2<T>)>
where
    T: NdFloat,
{
    let ss = step / T::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(a.shape()[0]);

    let be = (&eye - a * ss).inverse().expect("Could not invert matrix");

    let ab = be.dot(&(&eye + a * ss));
    let bb = (b * ss).dot(&b.t());

    Ok((ab, bb, c.clone()))
}

pub trait Discretize<T = f64>
where
    T: Float,
{
    type Output;

    fn discretize(&self, step: T) -> Self::Output;
}

pub enum DiscretizeArgs {}

pub struct Discretizer<T = f64> {
    pub step: T,
}

pub struct Discretized<T>(T);
