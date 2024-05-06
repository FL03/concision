/*
   Appellation: features <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{Dimension, ErrorKind, IntoDimension, Ix2, ShapeBuilder, ShapeError};


#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Features {
    pub inputs: usize,
    pub outputs: usize,
}

impl Features {
    pub fn new(outputs: usize, inputs: usize) -> Self {
        Self { inputs, outputs }
    }

    pub fn from_dimension<D>(dim: D) -> Result<Self, ShapeError>
    where
        D: Dimension,
    {
        if dim.ndim() == 1 {
            let res = Self::new(1, dim[0]);
            return Ok(res);
        } else if dim.ndim() >= 2 {
            let res = Self::new(dim[0], dim[1]);
            return Ok(res);
        }
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    }

    pub fn from_shape<D, Sh>(shape: Sh) -> Self
    where
        D: nd::RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape();
        let dim = shape.raw_dim().clone();
        Self::from_dimension(dim).expect("Invalid shape")
    }

    pub fn neuron(inputs: usize) -> Self {
        Self::new(1, inputs)
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn uniform_scale<T: num::Float>(&self) -> T {
        T::from(self.inputs()).unwrap().recip().sqrt()
    }
}

impl core::fmt::Display for Features {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {})", self.inputs, self.outputs)
    }
}

impl IntoDimension for Features {
    type Dim = Ix2;

    fn into_dimension(self) -> Self::Dim {
        ndarray::Ix2(self.outputs, self.inputs)
    }
}

impl TryFrom<nd::ArrayView1<'_, usize>> for Features {
    type Error = ShapeError;

    fn try_from(shape: nd::ArrayView1<'_, usize>) -> Result<Self, Self::Error> {
        if shape.len() == 1 {
            let tmp = Self {
                inputs: shape[0],
                outputs: 1,
            };
            return Ok(tmp);
        } else if shape.len() >= 2 {
            let tmp = Self {
                inputs: shape[1],
                outputs: shape[0],
            };
            return Ok(tmp);
        }
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    }
}

impl From<Features> for Ix2 {
    fn from(features: Features) -> Self {
        features.into_dimension()
    }
}

impl From<Features> for ndarray::IxDyn {
    fn from(features: Features) -> Self {
        ndarray::IxDyn(&[features.outputs, features.inputs])
    }
}

impl From<Features> for [usize; 2] {
    fn from(features: Features) -> Self {
        [features.outputs, features.inputs]
    }
}

impl From<[usize; 2]> for Features {
    fn from(features: [usize; 2]) -> Self {
        Self {
            inputs: features[1],
            outputs: features[0],
        }
    }
}

impl From<Features> for (usize, usize) {
    fn from(features: Features) -> Self {
        (features.outputs, features.inputs)
    }
}

impl From<(usize, usize)> for Features {
    fn from((inputs, outputs): (usize, usize)) -> Self {
        Self { inputs, outputs }
    }
}

impl From<usize> for Features {
    fn from(inputs: usize) -> Self {
        Self { inputs, outputs: 1 }
    }
}
