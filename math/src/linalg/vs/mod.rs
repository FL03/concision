/*
    Appellation: vs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Vector Space
//! 
//! A vector space is defined to be a set V, whose elements are called vectors, on which are defined two operations, 
//! called addition and multiplication by scalars (real numbers), subject to the ten axioms listed below.
//! 
//! ## Axioms
//! 
//! 1. Closure under addition
//! 2. Closure under scalar multiplication
//! 3. Commutativity of addition
//! 4. Associativity of addition
//! 5. Additive identity
//! 6. Additive inverse
//! 7. Distributivity of scalar multiplication with respect to vector addition
//! 8. Distributivity of scalar multiplication with respect to field addition
//! 9. Compatibility of scalar multiplication with field multiplication
//! 10. Identity element of scalar multiplication
pub use self::{space::*, utils::*};

pub(crate) mod space;



pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize, y: usize| x + y;
        let actual = f(4, 4);
        let expected: usize = 8;
        assert_eq!(actual, expected)
    }
}
