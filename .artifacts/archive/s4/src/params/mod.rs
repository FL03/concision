/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{dplr::*, kinds::*, store::*};

pub(crate) mod dplr;
pub(crate) mod kinds;
pub(crate) mod store;

use ndarray::prelude::{Array, Array2, Ix2};
use num::Num;
use std::collections::HashMap;

pub type SSMMap<T = f64, D = Ix2> = HashMap<SSMParams, Array<T, D>>;

pub trait SSMParamGroup<T = f64>
where
    T: Num,
{
    fn a(&self) -> &Array2<T>;
    fn b(&self) -> &Array2<T>;
    fn c(&self) -> &Array2<T>;
    fn d(&self) -> &Array2<T>;
}

impl<T> SSMParamGroup<T> for SSM<T>
where
    T: Num,
{
    fn a(&self) -> &Array2<T> {
        &self.a
    }

    fn b(&self) -> &Array2<T> {
        &self.b
    }

    fn c(&self) -> &Array2<T> {
        &self.c
    }

    fn d(&self) -> &Array2<T> {
        &self.d
    }
}

impl<T> SSMParamGroup<T> for SSMMap<T>
where
    T: Num,
{
    fn a(&self) -> &Array2<T> {
        self.get(&SSMParams::A).unwrap()
    }

    fn b(&self) -> &Array2<T> {
        self.get(&SSMParams::B).unwrap()
    }

    fn c(&self) -> &Array2<T> {
        self.get(&SSMParams::C).unwrap()
    }

    fn d(&self) -> &Array2<T> {
        self.get(&SSMParams::D).unwrap()
    }
}

impl<T> SSMParamGroup<T> for &[Array2<T>; 4]
where
    T: Num,
{
    fn a(&self) -> &Array2<T> {
        &self[0]
    }

    fn b(&self) -> &Array2<T> {
        &self[1]
    }

    fn c(&self) -> &Array2<T> {
        &self[2]
    }

    fn d(&self) -> &Array2<T> {
        &self[3]
    }
}

impl<T> SSMParamGroup<T> for (Array2<T>, Array2<T>, Array2<T>, Array2<T>)
where
    T: Num,
{
    fn a(&self) -> &Array2<T> {
        &self.0
    }

    fn b(&self) -> &Array2<T> {
        &self.1
    }

    fn c(&self) -> &Array2<T> {
        &self.2
    }

    fn d(&self) -> &Array2<T> {
        &self.3
    }
}

#[cfg(test)]
mod tests {}
