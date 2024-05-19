/*
   Appellation: binary <activate>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{Array, ArrayBase, Data, Dimension};
use num::{One, Zero};

///
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

unary!(Heavyside::heavyside(self),);

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
        }
    };
}

impl_heavyside!(f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,);

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
}

impl<'a, A, B, S, D> Heavyside for &'a ArrayBase<S, D>
where
    A: Clone + Heavyside<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(Heavyside::heavyside)
    }
}
