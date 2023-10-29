/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::IntoDimension;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Features {
   pub input: usize,
   pub output: usize,
}

impl Features {
   pub fn new(input: usize, output: usize) -> Self {
      Self { input, output }
   }

   pub fn input(&self) -> usize {
      self.input
   }

   pub fn output(&self) -> usize {
      self.output
   }
}

impl IntoDimension for Features {
   type Dim = ndarray::IxDyn;

   fn into_dimension(self) -> Self::Dim {
      ndarray::IxDyn(&[self.input, self.output])
   }
}
