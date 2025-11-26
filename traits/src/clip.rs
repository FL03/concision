/*
    Appellation: clip <module>
    Contrib: @FL03
*/

/// A trait denoting objects capable of being _clipped_ between some minimum and some maximum.
pub trait Clip<T> {
    /// the type output produced by the operation
    type Output;
    /// limits the values within an object to a range between `min` and `max`
    fn clip(&self, min: T, max: T) -> Self::Output;
}

/// This trait enables tensor clipping; it is implemented for `ArrayBase`
pub trait ClipMut<T = f32> {
    /// clip the tensor between the minimum and maximum values
    fn clip_between(&mut self, min: T, max: T);

    fn clip_inf_nan(&mut self, on_inf: T, on_nan: T);
    /// clip the tensor between a boundary value, replacing any infinite or NaN values
    fn clip_inf_nan_between(&mut self, boundary: T, on_inf: T, on_nan: T);
    /// clip any infinite values in the tensor
    fn clip_inf(&mut self, threshold: T);
    /// clip the tensor to a maximum threshold
    fn clip_max(&mut self, threshold: T);
    /// clip the tensor to a minimum threshold
    fn clip_min(&mut self, threshold: T);
    /// this method normalizes the tensor then clips any values outside of the given threshold.
    /// the tensor is normalized using the L1 norm
    fn clip_norm_l1(&mut self, threshold: T);
    /// this method normalizes the tensor then clips any values outside of the given threshold.
    /// the tensor is normalized using the L2 norm
    fn clip_norm_l2(&mut self, threshold: T);
    /// clip any NaN values in the tensor
    fn clip_nan(&mut self, threshold: T);
}

/*
 ************* Implementations *************
*/
use crate::norm::{L1Norm, L2Norm};
use ndarray::{ArrayBase, Dimension, ScalarOperand};
use num_traits::Float;

impl<A, S, D> Clip<A> for ArrayBase<S, D, A>
where
    A: 'static + Clone + PartialOrd,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    type Output = ndarray::Array<A, D>;

    fn clip(&self, min: A, max: A) -> Self::Output {
        self.clamp(min, max)
    }
}

impl<A, S, D> ClipMut<A> for ArrayBase<S, D, A>
where
    A: Float + ScalarOperand,
    S: ndarray::DataMut<Elem = A>,
    D: Dimension,
{
    fn clip_between(&mut self, min: A, max: A) {
        self.mapv_inplace(|x| {
            if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            }
        });
    }

    fn clip_inf_nan(&mut self, on_inf: A, on_nan: A) {
        self.mapv_inplace(|x| {
            if x.is_nan() {
                on_nan
            } else if x.is_infinite() {
                on_inf
            } else {
                x
            }
        });
    }

    fn clip_inf_nan_between(&mut self, boundary: A, on_inf: A, on_nan: A) {
        self.mapv_inplace(|x| {
            if x.is_nan() {
                on_nan
            } else if x.is_infinite() {
                on_inf
            } else if x < boundary.neg() {
                boundary.neg()
            } else if x > boundary {
                boundary
            } else {
                x
            }
        });
    }

    fn clip_inf(&mut self, threshold: A) {
        self.mapv_inplace(|x| if x.is_infinite() { threshold } else { x });
    }

    fn clip_max(&mut self, threshold: A) {
        self.mapv_inplace(|x| if x > threshold { threshold } else { x });
    }

    fn clip_min(&mut self, threshold: A) {
        self.mapv_inplace(|x| if x < threshold { threshold } else { x });
    }

    fn clip_nan(&mut self, threshold: A) {
        self.mapv_inplace(|x| if x.is_nan() { threshold } else { x });
    }

    fn clip_norm_l1(&mut self, threshold: A) {
        let norm = self.l1_norm();
        if norm > threshold {
            self.mapv_inplace(|x| x * threshold / norm);
        }
    }

    fn clip_norm_l2(&mut self, threshold: A) {
        let norm = self.l2_norm();
        if norm > threshold {
            self.mapv_inplace(|x| x * threshold / norm);
        }
    }
}
