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
    use nd::prelude::{Array, ArrayBase, ArrayView, Axis, Dimension};
    use nd::{Data, ScalarOperand};
    use num::complex::ComplexFloat;

    pub(crate) fn scale<A>(dk: usize) -> A
    where
        A: ComplexFloat,
    {
        A::from(dk).unwrap().sqrt().recip()
    }

    /// Scaled dot-product attention;
    pub fn scaled_dot_product_attention<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
    ) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let dk = scale::<A>(k.len_of(Axis(1)));
        (q.dot(&k.t()) * dk).softmax().dot(&v)
    }
}
