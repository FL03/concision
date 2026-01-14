/*
    Appellation: shape <module>
    Created At: 2025.12.14:11:03:05
    Contrib: @FL03
*/
use ndarray::RemoveAxis;

pub trait GetBiasDim<A, D>
where
    D: RemoveAxis,
{
    type Output;

    fn get_bias_dim(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/

impl<A, D, U> GetBiasDim<A, D> for U
where
    D: RemoveAxis,
    U: AsRef<ndarray::LayoutRef<A, D>>,
{
    type Output = D::Smaller;

    fn get_bias_dim(&self) -> Self::Output {
        crate::extract_bias_dim(self)
    }
}
