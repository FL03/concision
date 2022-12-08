use concision::num::complex::{C, Complex};
use std::ops::Mul;

// Define a holomorphic function that squares its input.
fn square<T: Complex + Mul<Output = T> + Clone>(z: T) -> T {
    z.clone() * z
}

fn main() {
    let c = C::from((1.0, 1.0));
    let res = square(c);
    assert_eq!(res.clone(), C::from((0.0, 2.0)));
    
    println!("{:?}", res);
}
