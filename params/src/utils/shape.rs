/*
    Appellation: shape <module>
    Created At: 2025.12.14:07:37:48
    Contrib: @FL03
*/
use ndarray::{Axis, LayoutRef, RemoveAxis};

/// Extract a suitable dimension for a bias tensor from the given reference to the layout of
/// the weight tensor.
pub fn extract_bias_dim<A, D>(layout: impl AsRef<LayoutRef<A, D>>) -> D::Smaller
where
    D: RemoveAxis,
{
    let layout = layout.as_ref();
    let dim = layout.raw_dim();
    dim.remove_axis(Axis(0))
}

#[cfg(test)]
mod tests {
    use super::extract_bias_dim;
    use ndarray::{Array, array};

    #[test]
    fn test_extract_bias_dim() {
        let layout = Array::linspace(0f32, 1f32, 100);
        let bias_dim = extract_bias_dim(&layout);
        assert_eq!(bias_dim, ndarray::Ix0());

        let layout = array![[1., 2., 3.], [4., 5., 6.]];
        let bias_dim = extract_bias_dim(&layout);
        assert_eq!(bias_dim, ndarray::Ix1(3));
    }
}
