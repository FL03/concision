/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, init::*, math::*, storage::*};

pub(crate) mod arrays;
pub(crate) mod init;
pub(crate) mod math;
pub(crate) mod storage;

use ndarray::prelude::{Array, Dimension};

pub trait Apply<T> {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
}

impl<T, D> Apply<T> for Array<T, D>
where
    D: Dimension,
{
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        self.map(f)
    }
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub(crate) mod utils {
    use ndarray::prelude::{s, Array2};
    use ndarray::ScalarOperand;
    use num::traits::{Num, NumAssignOps};

    pub fn inverse<T>(matrix: &Array2<T>) -> Option<Array2<T>>
    where
        T: Copy + Num + NumAssignOps + ScalarOperand,
    {
        let (rows, cols) = matrix.dim();

        if !matrix.is_square() {
            return None; // Matrix must be square for inversion
        }

        let identity = Array2::eye(rows);

        // Construct an augmented matrix by concatenating the original matrix with an identity matrix
        let mut aug = Array2::zeros((rows, 2 * cols));
        aug.slice_mut(s![.., ..cols]).assign(matrix);
        aug.slice_mut(s![.., cols..]).assign(&identity);

        // Perform Gaussian elimination to reduce the left half to the identity matrix
        for i in 0..rows {
            let pivot = aug[[i, i]];

            if pivot == T::zero() {
                return None; // Matrix is singular
            }

            aug.slice_mut(s![i, ..]).mapv_inplace(|x| x / pivot);

            for j in 0..rows {
                if i != j {
                    let am = aug.clone();
                    let factor = aug[[j, i]];
                    let rhs = am.slice(s![i, ..]);
                    aug.slice_mut(s![j, ..])
                        .zip_mut_with(&rhs, |x, &y| *x -= y * factor);
                }
            }
        }

        // Extract the inverted matrix from the augmented matrix
        let inverted = aug.slice(s![.., cols..]);

        Some(inverted.to_owned())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_affine() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];

        let y = x.affine(4.0, -2.0).unwrap();
        assert_eq!(y, array![[-2.0, 2.0], [6.0, 10.0]]);
    }

    #[test]
    fn test_matrix_power() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(x.pow(0), Array2::<f64>::eye(2));
        assert_eq!(x.pow(1), x);
        assert_eq!(x.pow(2), x.dot(&x));
    }
}
