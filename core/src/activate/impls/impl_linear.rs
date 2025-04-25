/*
    Appellation: linear <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

impl<'a, T> crate::activate::LinearActivation for &'a T
where
    T: Clone + Default,
{
    type Output = T;

    fn linear(self) -> Self::Output {
        self.clone()
    }

    fn linear_derivative(self) -> Self::Output {
        <T>::default()
    }
}
