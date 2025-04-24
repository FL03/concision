/*
    Appellation: dropout <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

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

/*
 ************* Implementations *************
*/

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(feature = "rand")]
mod impls {
    use super::Dropout;

    use concision_core::{Forward, init::InitializeExt};
    use ndarray::{Array, ArrayBase, DataOwned, Dimension, ScalarOperand};
    use num::traits::Num;

    impl Dropout {
        pub fn apply<A, S, D>(&self, input: &ArrayBase<S, D>) -> Array<A, D>
        where
            A: Num + ScalarOperand,
            D: Dimension,
            S: DataOwned<Elem = A>,
        {
            let Dropout { p } = *self;
            let dim = input.dim();

            // Create a mask of the same shape as the input array
            let mask = {
                let tmp = Array::<bool, D>::bernoulli(dim, p).expect("Failed to create mask");
                tmp.mapv(|x| if x { A::zero() } else { A::one() })
            };

            // Element-wise multiplication to apply dropout
            input.to_owned() * mask
        }
    }

    impl<A, S, D> Forward<ArrayBase<S, D>> for Dropout
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        type Output = Array<A, D>;

        fn forward(&self, input: &ArrayBase<S, D>) -> cnc::Result<Self::Output> {
            let res = self.apply(input);
            Ok(res)
        }
    }
}
