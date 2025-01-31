/*
    Appellation: avg <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision_math::Squared;
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num::traits::{FromPrimitive, Num, Pow, Signed};

pub trait MeanAbsoluteError<Rhs = Self> {
    type Output;

    fn mae(&self, target: &Rhs) -> Self::Output;
}

pub trait MeanSquaredError<Rhs = Self> {
    type Output;

    fn mse(&self, target: &Rhs) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<A, S, D> MeanAbsoluteError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num + ScalarOperand + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn mae(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (target - self).mean().unwrap().abs()
        // let total_error = self
        //     .into_iter()
        //     .zip(target)
        //     .fold(A::zero(), |acc, (&a, &b)| acc + (a - b).abs());
        // total_error / A::from_usize(self.len()).unwrap()
    }
}

impl<A, S, D> MeanSquaredError<ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: FromPrimitive + Num + Pow<i32, Output = A> + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Option<A>;

    fn mse(&self, target: &ArrayBase<S, D>) -> Self::Output {
        (target - self).sqrd().mean()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mae_loss() {
        let pred = array![1.0, 2.0, 3.0, 4.0];
        let actual = array![1.0, 3.0, 3.5, 4.5];
        assert_eq!(pred.mae(&actual), 0.5);
    }
}
