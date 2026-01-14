/*
    Appellation: shuffle <module>
    Contrib: @FL03
*/
#![cfg(feature = "rand")]

use rand::RngCore;

/// The [`Shuffle`] trait establishes an interface for randomly shuffling collections of
/// elements using a provided random number generator.
pub trait Shuffle<R: RngCore> {
    fn shuffle(&mut self, rng: &mut R);
}

/*
 ************* Implementations *************
*/
use rand::Rng;

impl<R, T> Shuffle<R> for [T]
where
    R: RngCore,
{
    fn shuffle(&mut self, rng: &mut R) {
        for i in (1..self.len()).rev() {
            self.swap(i, rng.random_range(0..=i));
        }
    }
}

impl<const N: usize, R, T> Shuffle<R> for [T; N]
where
    R: RngCore,
{
    fn shuffle(&mut self, rng: &mut R) {
        for i in (1..self.len()).rev() {
            self.swap(i, rng.random_range(0..=i));
        }
    }
}

#[cfg(feature = "alloc")]
impl<R, T> Shuffle<R> for alloc::vec::Vec<T>
where
    R: RngCore,
{
    fn shuffle(&mut self, rng: &mut R) {
        for i in (1..self.len()).rev() {
            self.swap(i, rng.random_range(0..=i));
        }
    }
}

// #[cfg(feature = "ndarray")]
// impl<S, D, A, R> Shuffle<R> for ndarray::ArrayBase<S, D, A>
// where
//     S: ndarray::DataMut<Elem = A>,
//     D: ndarray::Dimension,
//     R: RngCore,
// {
//     fn shuffle(&mut self, rng: &mut R) {
//         use rand::Rng;
//         let len = self.len();
//         for i in (1..len).rev() {
//             let j = rng.random_range(0..=i);
//             self.swap([i], [j]);
//         }
//     }
// }
