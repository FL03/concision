/*
    Appellation: mask <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use nd::prelude::*;
use nd::RawData;

pub trait NdMask<D = Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = bool>;
}

pub struct Mask<S, D>(ArrayBase<S, D>)
where
    D: Dimension,
    S: RawData<Elem = bool>;

impl<S, D> Mask<S, D>
where
    D: Dimension,
    S: RawData<Elem = bool>,
{
    pub fn new(data: ArrayBase<S, D>) -> Self {
        Self(data)
    }
}

/*
 ************* Implementations *************
*/
mod impls {
    use super::*;
    use core::borrow::{Borrow, BorrowMut};
    use core::ops::{Deref, DerefMut};

    impl<S, D> AsRef<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn as_ref(&self) -> &ArrayBase<S, D> {
            &self.0
        }
    }

    impl<S, D> AsMut<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn as_mut(&mut self) -> &mut ArrayBase<S, D> {
            &mut self.0
        }
    }

    impl<S, D> Borrow<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn borrow(&self) -> &ArrayBase<S, D> {
            &self.0
        }
    }

    impl<S, D> BorrowMut<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn borrow_mut(&mut self) -> &mut ArrayBase<S, D> {
            &mut self.0
        }
    }

    impl<S, D> Deref for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        type Target = ArrayBase<S, D>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<S, D> DerefMut for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl<S, D> From<ArrayBase<S, D>> for Mask<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn from(mask: ArrayBase<S, D>) -> Self {
            Mask(mask)
        }
    }

    impl<S, D> From<Mask<S, D>> for ArrayBase<S, D>
    where
        D: Dimension,
        S: RawData<Elem = bool>,
    {
        fn from(mask: Mask<S, D>) -> Self {
            mask.0
        }
    }
}
