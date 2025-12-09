/*
    Appellation: dimensionality <module>
    Created At: 2025.12.09:10:03:43
    Contrib: @FL03
*/

/// the [`Dim`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait Dim {
    type Shape;

    private! {}

    /// returns the rank of the dimension; the rank essentially speaks to the total number of 
    /// axes defined by the dimension.
    fn rank(&self) -> usize;
    /// returns the total number of elements considered by the dimension
    fn size(&self) -> usize;
}

/*
    ************* Implementations *************
*/

impl<D> Dim for D
where
    D: nd::Dimension,
{
    type Shape = D::Pattern;

    seal! {}

    fn rank(&self) -> usize {
        self.ndim()
    }

    fn size(&self) -> usize {
        self.size()
    }
}