/*
   Appellation: ops <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{dropout::*, norm::*, utils::*};

pub(crate) mod dropout;
pub(crate) mod norm;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Forward;
    use concision_core::prelude::RoundTo;
    use ndarray::prelude::{array, Array, Ix2};

    #[test]
    fn test_layer_norm() {
        let features = 4;
        let data: Array<f64, Ix2> = Array::linspace(1., 4., 4)
            .into_shape((1, features))
            .unwrap();
        let norm = LayerNorm::<f64>::new((1, features));
        let normed = norm.forward(&data);
        let rounded = normed.map(|x| x.round_to(4));
        let exp = array![[-1.1619, -0.3873, 0.3873, 1.1619]];
        assert_eq!(rounded, exp);
    }
}
