/*
    Appellation: dropout <module>
    Created At: 2025.11.26:17:01:56
    Contrib: @FL03
*/

/// [Dropout] randomly zeroizes elements with a given probability (`p`).
pub trait DropOut {
    type Output;

    fn dropout(&self, p: f64) -> Self::Output;
}

/// The [Dropout] layer is randomly zeroizes inputs with a given probability (`p`).
/// This regularization technique is often used to prevent overfitting.
///
///
/// ### Config
///
/// - (p) Probability of dropping an element
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Dropout {
    pub(crate) p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }

    pub fn forward<U>(&self, input: &U) -> Option<<U as DropOut>::Output>
    where
        U: DropOut,
    {
        Some(input.dropout(self.p))
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(feature = "rand")]
mod impl_rand {
    use super::*;
    use concision_init::NdRandom;
    use concision_traits::Forward;
    use ndarray::{Array, ArrayBase, DataOwned, Dimension, ScalarOperand};
    use num_traits::Num;

    impl<A, S, D> DropOut for ArrayBase<S, D, A>
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        type Output = Array<A, D>;

        fn dropout(&self, p: f64) -> Self::Output {
            let dim = self.dim();
            // Create a mask of the same shape as the input array
            let mask: Array<bool, D> = Array::bernoulli(dim, p).expect("Failed to create mask");
            let mask = mask.mapv(|x| if x { A::zero() } else { A::one() });

            // Element-wise multiplication to apply dropout
            self.to_owned() * mask
        }
    }

    impl<U> Forward<U> for Dropout
    where
        U: DropOut,
    {
        type Output = <U as DropOut>::Output;

        fn forward(&self, input: &U) -> Self::Output {
            input.dropout(self.p)
        }
    }
}

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dropout() {
        let shape = (512, 2048);
        let arr = Array2::<f64>::ones(shape);
        let dropout = Dropout::new(0.5);
        let out = dropout.forward(&arr).expect("Dropout forward pass failed");

        assert!(arr.iter().all(|&x| x == 1.0));
        assert!(out.iter().any(|x| x == &0f64));
    }
}
