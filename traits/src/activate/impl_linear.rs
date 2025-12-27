/*
    Appellation: impl_linear <module>
    Created At: 2025.12.14:11:14:22
    Contrib: @FL03
*/
use super::{HeavysideActivation, LinearActivation};
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};
use num_traits::{One, Zero};

macro_rules! impl_heavyside {
    ($($T:ty),* $(,)*) => {
        $(
            impl HeavysideActivation for $T {
                type Output = $T;

                fn heavyside(self) -> Self::Output {
                    if self > <$T>::zero() {
                        <$T>::one()
                    } else {
                        <$T>::zero()
                    }
                }

                fn heavyside_derivative(self) -> Self::Output {
                    if self > <$T>::zero() {
                        <$T>::one()
                    } else {
                        <$T>::zero()
                    }
                }
            }
        )*
    };
}

macro_rules! impl_linear {
    ($($T:ty),* $(,)*) => {
        $(
            impl LinearActivation for $T {
                type Output = $T;

                fn linear(self) -> Self::Output {
                    self
                }

                fn linear_derivative(self) -> Self::Output {
                    <$T>::one()
                }
            }
        )*
    };
}

impl_heavyside!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
);

impl_linear!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
);

impl<A, B, S, D> HeavysideActivation for ArrayBase<S, D, A>
where
    A: Clone + HeavysideActivation<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside_derivative)
    }
}

impl<A, B, S, D> HeavysideActivation for &ArrayBase<S, D, A>
where
    A: Clone + HeavysideActivation<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside_derivative)
    }
}

impl<A, S, D> LinearActivation for ArrayBase<S, D, A>
where
    A: Clone + One,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Output = ArrayBase<S, D, A>;

    fn linear(self) -> Self::Output {
        self
    }

    fn linear_derivative(self) -> Self::Output {
        self.mapv_into(|_| <A>::one())
    }
}
