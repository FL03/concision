/*
    appellation: convert <module>
    authors: @FL03
*/
use ndarray::Axis;

/// The [`IntoAxis`] trait is used to define a conversion routine that takes a type and wraps
/// it in an [`Axis`] type.
pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

/*
 ************* Implementations *************
*/

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}
