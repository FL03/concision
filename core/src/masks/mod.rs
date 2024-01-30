/*
   Appellation: masks <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Mask
pub use self::{mask::*, utils::*};

pub(crate) mod mask;

pub trait Masked<T> {
    fn mask(&self) -> &Mask<T>;
    fn mask_mut(&mut self) -> &mut Mask<T>;
}

pub(crate) mod utils {
    use super::Mask;
    use ndarray::prelude::Array2;
    use ndarray_rand::rand_distr::uniform::SampleUniform;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use num::Float;

    pub fn rmask_uniform<T>(features: usize) -> Mask<T>
    where
        T: Float + SampleUniform,
    {
        let ds = (T::one() / T::from(features).unwrap()).sqrt();
        Mask::from(Array2::<T>::random(
            (features, features),
            Uniform::new(-ds, ds),
        ))
    }
}

#[cfg(test)]
mod tests {}
