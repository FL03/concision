/*
    Appellation: linear <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub fn linear<T>(x: T) -> T {
    x
}

unary!(LinearActivation::linear(self));

impl<'a, T> LinearActivation for &'a T
where
    T: Clone,
{
    type Output = T;

    fn linear(self) -> Self::Output {
        self.clone()
    }
}
