/*
   Appellation: pad <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::{action::*, error::*, mode::*, utils::*};

pub(crate) mod action;
pub(crate) mod error;
pub(crate) mod mode;
pub(crate) mod utils;

use nd::{Array, ArrayBase, DataOwned, Dimension};
use num::traits::{FromPrimitive, Num};

pub trait Pad<T> {
    type Output;

    fn pad(&self, mode: PadMode<T>, pad: &[[usize; 2]]) -> Self::Output;
}

impl<A, S, D> Pad<A> for ArrayBase<S, D>
where
    A: Copy + FromPrimitive + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn pad(&self, mode: PadMode<A>, pad: &[[usize; 2]]) -> Self::Output {
        utils::pad(self, pad, mode).unwrap()
    }
}

pub struct Padding<T> {
    pub(crate) action: PadAction,
    pub(crate) mode: PadMode<T>,
    pub(crate) pad: Vec<[usize; 2]>,
    pub(crate) padding: usize,
}

impl<T> Padding<T> {
    pub fn new() -> Self {
        Self {
            action: PadAction::default(),
            mode: PadMode::default(),
            pad: Vec::new(),
            padding: 0,
        }
    }

    pub fn pad(&self) -> &[[usize; 2]] {
        &self.pad
    }

    pub fn with_action(mut self, action: PadAction) -> Self {
        self.action = action;
        self
    }

    pub fn with_mode(mut self, mode: PadMode<T>) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }
}
