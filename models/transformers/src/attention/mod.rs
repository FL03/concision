/*
    Appellation: attention <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::head::AttentionHead;
pub use self::utils::*;

pub(crate) mod head;

pub mod multi;

pub(crate) mod prelude {
    pub use super::head::AttentionHead;
    pub use super::utils::*;
}

pub(crate) mod utils {
    use concision::func::activate::Softmax;
    use nd::linalg::Dot;
    use nd::{Array, Axis, Dimension, ScalarOperand};
    use num::complex::ComplexFloat;

    pub(crate) fn scale_dk<A>(dk: A) -> A
    where
        A: ComplexFloat + ScalarOperand,
    {
        dk.sqrt().recip()
    }

    pub fn scaled_dot_product<A, D>(
        q: &Array<A, D>,
        k: &Array<A, D>,
        v: &Array<A, D>,
    ) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        D: Dimension,
        Array<A, D>: Dot<Array<A, D>, Output = Array<A, D>>,
    {
        let qk = q.dot(&k.t().to_owned());
        let scale = scale_dk(A::from(k.len_of(Axis(1))).unwrap());
        let scaled = qk * scale.recip();
        scaled.softmax().dot(&v)
    }
}
