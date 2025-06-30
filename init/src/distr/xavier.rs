/*
    Appellation: xavier <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Xavier
//!
//! Xavier initialization techniques were developed in 2010 by Xavier Glorot.
//! These methods are designed to initialize the weights of a neural network in a way that
//! prevents the vanishing and exploding gradient problems. The initialization technique
//! manifests into two distributions: [XavierNormal] and [XavierUniform].
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Distribution, StandardNormal};

/// Normal Xavier initializers leverage a normal distribution centered around `0` and using a
/// standard deviation ($\sigma$) computed by:
///
/// ```math
/// \sigma = \sqrt{\frac{2}{d_{in} + d_{out}}}
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct XavierNormal<T>
where
    StandardNormal: Distribution<T>,
{
    std: T,
}

/// Uniform Xavier initializers use a uniform distribution to initialize the weights of a neural network
/// within a given range.
pub struct XavierUniform<T>
where
    T: SampleUniform,
{
    distr: Uniform<T>,
}

/*
 ************* Implementations *************
*/

mod impl_normal {
    use super::XavierNormal;
    use num_traits::{Float, FromPrimitive};
    use rand::RngCore;
    use rand_distr::{Distribution, Normal, StandardNormal};

    fn std_dev<T>(inputs: usize, outputs: usize) -> T
    where
        T: FromPrimitive + Float,
    {
        let numerator = T::from_usize(2).unwrap();
        let denominator = T::from_usize(inputs + outputs).unwrap();
        (numerator / denominator).sqrt()
    }

    impl<T> XavierNormal<T>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        pub fn new(inputs: usize, outputs: usize) -> Self
        where
            T: FromPrimitive,
        {
            Self {
                std: std_dev(inputs, outputs),
            }
        }
        /// tries creating a new [`Normal`] distribution with a mean of 0 and the computed
        /// standard deviation ($\sigma$) based on the number of inputs and outputs.
        pub fn distr(&self) -> crate::Result<Normal<T>> {
            Normal::new(T::zero(), self.std_dev()).map_err(Into::into)
        }
        /// returns a reference to the standard deviation of the distribution
        pub const fn std_dev(&self) -> T {
            self.std
        }
    }

    impl<T> Distribution<T> for XavierNormal<T>
    where
        T: Float,
        StandardNormal: Distribution<T>,
    {
        fn sample<R>(&self, rng: &mut R) -> T
        where
            R: RngCore + ?Sized,
        {
            self.distr().unwrap().sample(rng)
        }
    }
}

mod impl_uniform {
    use super::XavierUniform;
    use num_traits::{Float, FromPrimitive};
    use rand::RngCore;
    use rand_distr::Distribution;
    use rand_distr::uniform::{SampleUniform, Uniform};

    fn boundary<U>(inputs: usize, outputs: usize) -> U
    where
        U: FromPrimitive + Float,
    {
        let numer = <U>::from_usize(6).unwrap();
        let denom = <U>::from_usize(inputs + outputs).unwrap();
        (numer / denom).sqrt()
    }

    impl<T> XavierUniform<T>
    where
        T: SampleUniform,
    {
        pub fn new(inputs: usize, outputs: usize) -> crate::Result<Self>
        where
            T: Float + FromPrimitive,
        {
            // calculate the boundary for the uniform distribution
            let limit = boundary::<T>(inputs, outputs);
            // create a uniform distribution with the calculated limit
            let distr = Uniform::new(-limit, limit)?;
            Ok(Self { distr })
        }
        /// returns an immutable reference to the underlying uniform distribution
        pub(crate) const fn distr(&self) -> &Uniform<T> {
            &self.distr
        }
    }

    impl<T> Distribution<T> for XavierUniform<T>
    where
        T: Float + SampleUniform,
    {
        fn sample<R>(&self, rng: &mut R) -> T
        where
            R: RngCore + ?Sized,
        {
            self.distr().sample(rng)
        }
    }

    impl<T> Clone for XavierUniform<T>
    where
        T: Clone + SampleUniform,
        <T as SampleUniform>::Sampler: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                distr: self.distr.clone(),
            }
        }
    }

    impl<T> Copy for XavierUniform<T>
    where
        T: Copy + SampleUniform,
        <T as SampleUniform>::Sampler: Copy,
    {
    }

    impl<T> Eq for XavierUniform<T>
    where
        T: Eq + SampleUniform,
        <T as SampleUniform>::Sampler: Eq,
    {
    }

    impl<T> PartialEq for XavierUniform<T>
    where
        T: PartialEq + SampleUniform,
        <T as SampleUniform>::Sampler: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.distr == other.distr
        }
    }
}
