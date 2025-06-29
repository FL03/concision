/*
   Appellation: binary <activate>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::activate::{Heavyside, utils::heavyside};
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::{One, Zero};

macro_rules! impl_heavyside {
    ($($ty:ty),* $(,)*) => {
        $(impl_heavyside!(@impl $ty);)*
    };
    (@impl $ty:ty) => {
        impl Heavyside for $ty {
            type Output = $ty;

            fn heavyside(self) -> Self::Output {
                heavyside(self)
            }

            fn heavyside_derivative(self) -> Self::Output {
                if self > <$ty>::zero() {
                    <$ty>::one()
                } else {
                    <$ty>::zero()
                }
            }
        }
    };
}

impl_heavyside!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,
);

impl<A, B, S, D> Heavyside for ArrayBase<S, D>
where
    A: Clone + Heavyside<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(Heavyside::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(Heavyside::heavyside_derivative)
    }
}

impl<A, B, S, D> Heavyside for &ArrayBase<S, D>
where
    A: Clone + Heavyside<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(Heavyside::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(Heavyside::heavyside_derivative)
    }
}
