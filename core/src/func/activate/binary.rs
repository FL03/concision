/*
   Appellation: binary <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::{One, Zero};

pub fn heavyside<T>(x: &T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > &T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

build_unary_trait!(Heavyside.heavyside,);