/*
    Appellation: binary <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension};
use num::{One, Zero};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Heavyside;

impl Heavyside {
    pub fn new() -> Self {
        Self
    }

    pub fn heavyside<T>(args: &T) -> T
    where
        T: One + PartialOrd + Zero,
    {
        if args > &T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}

// impl<T, D> Activate<T, D> for Heavyside
// where
//     D: Dimension,
//     T: Clone + One + PartialOrd + Zero,
// {
//     fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
//         args.mapv(|x| Self::heavyside(&x))
//     }
// }

// impl<T, D> FnOnce<(&Array<T, D>,)> for Heavyside
// where
//     D: Dimension,
//     T: Clone + One + PartialOrd + Zero,
// {
//     type Output = Array<T, D>;

//     extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Array<T, D> {
//         args.mapv(|x| Self::heavyside(&x))
//     }
// }

impl<T, D> Fn<(&Array<T, D>,)> for Heavyside
where
    D: Dimension,
    T: One + PartialOrd + Zero,
{
    extern "rust-call" fn call(&self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Heavyside::heavyside)
    }
}

impl<T, D> FnMut<(&Array<T, D>,)> for Heavyside
where
    D: Dimension,
    T: One + PartialOrd + Zero,
{
    extern "rust-call" fn call_mut(&mut self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Heavyside::heavyside)
    }
}

impl<T, D> FnOnce<(&Array<T, D>,)> for Heavyside
where
    D: Dimension,
    T: One + PartialOrd + Zero,
{
    type Output = Array<T, D>;

    extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Self::Output {
        args.0.map(Heavyside::heavyside)
    }
}
