/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Records<T> {
    fn features(&self) -> usize;

    fn samples(&self) -> usize;
}

impl<S, T> Records<T> for S
where
    S: AsRef<(usize, usize)>,
{
    fn features(&self) -> usize {
        self.as_ref().1
    }

    fn samples(&self) -> usize {
        self.as_ref().0
    }
}

pub trait NdArrayExt<T> {}
