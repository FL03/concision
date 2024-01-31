/*
   Appellation: shapes <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Shapes
pub use self::{dimension::*, rank::*, shape::*};

pub(crate) mod dimension;
pub(crate) mod rank;
pub(crate) mod shape;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let mut shape = Shape::default();
        shape.extend([1, 1, 1]);
        assert_eq!(shape, Shape::new(vec![1, 1, 1]));
        assert_eq!(shape.elements(), 1);
        assert_eq!(*shape.rank(), 3);
    }
}
