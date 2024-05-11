/*
    Appellation: impl_linalg <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::QKVBase;
use concision::Matmul;
use nd::linalg::Dot;
use nd::*;

impl<A, S, D, E> Matmul<QKVBase<S, E>> for QKVBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: Dot<ArrayBase<S, E>, Output = ArrayBase<S, E>>,
{
    type Output = QKVBase<S, E>;

    fn matmul(&self, rhs: &QKVBase<S, E>) -> Self::Output {
        QKVBase {
            q: self.q().dot(rhs.q()),
            k: self.k().dot(rhs.k()),
            v: self.v().dot(rhs.v()),
        }
    }
}

impl<A, S, D, E> Matmul<ArrayBase<S, E>> for QKVBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    S: RawData<Elem = A>,
    ArrayBase<S, D>: Dot<ArrayBase<S, E>, Output = ArrayBase<S, E>>,
{
    type Output = QKVBase<S, E>;

    fn matmul(&self, rhs: &ArrayBase<S, E>) -> Self::Output {
        QKVBase {
            q: self.q().dot(rhs),
            k: self.k().dot(rhs),
            v: self.v().dot(rhs),
        }
    }
}

