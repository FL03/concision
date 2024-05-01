/*
   Appellation: convert <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::Axis;

pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

/*
 ******** implementations ********
*/

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}
