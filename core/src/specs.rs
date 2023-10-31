/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::{Num, One};
use std::ops::MulAssign;

pub trait Pair<A, B> {
    fn pair(&self) -> (A, B);
}

impl<A, B, T> Pair<A, B> for T
where
    T: Clone + Into<(A, B)>,
{
    fn pair(&self) -> (A, B) {
        self.clone().into()
    }
}

pub trait Product {
    type Item: Num;

    fn product(&self) -> Self::Item;
}

impl<I, T> Product for I
where
    I: Clone + IntoIterator<Item = T>,
    T: One + Num + MulAssign<T>,
{
    type Item = T;

    fn product(&self) -> Self::Item {
        let mut res = T::one();
        for i in self.clone().into_iter() {
            res *= i;
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product() {
        let args = vec![2, 4, 6];
        assert_eq!(args.product(), 48);
    }
}
