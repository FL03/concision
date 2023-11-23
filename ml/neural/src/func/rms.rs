/*
   Appellation: rms <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension};
use num::{Float, FromPrimitive};

pub fn rms<T, D>(args: &Array<T, D>) -> Option<T>
where
    D: Dimension,
    T: Float + FromPrimitive,
{
    if let Some(avg) = args.mapv(|xs| xs.powi(2)).mean() {
        return Some(avg.sqrt());
    }
    None
}

pub trait RootMeanSquare<T> {
    fn rms(&self) -> T;
}

// impl<S, T> RootMeanSquare<T> for S where S: ExactSizeIterator<Item = T>, T: Float {
//     fn rms(&self) -> T {
//         let sum = self.iter().fold(T::zero(), |acc, x| acc + x.powi(2));
//         (sum / T::from(self.len()).unwrap()).sqrt()
//     }
// }

impl<T, D> RootMeanSquare<T> for Array<T, D>
where
    D: Dimension,
    T: Float + FromPrimitive,
{
    fn rms(&self) -> T {
        (self.mapv(|xs| xs.powi(2)).mean().unwrap_or_else(T::zero)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use computare::prelude::RoundTo;
    use ndarray::prelude::Ix1;

    #[test]
    fn test_rms() {
        let v = linarr::<f64, Ix1>(5).expect("Failed to create array");
        let rms = v.rms().round_to(5);
        assert_eq!(rms, 3.31662);
    }
}
