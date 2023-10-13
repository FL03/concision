extern crate concision;

use concision::math::num::complex::C;
use concision::prelude::Numerical;

// Define a holomorphic function that squares its input.
fn square<T: Numerical>(z: C<T>) -> C<T> {
    z.clone() * z
}

fn main() {
    let c = C::from((1.0, 1.0));
    let res = square(c);
    assert_eq!(res.clone(), C::from((0.0, 2.0)));

    println!("{:?}", res);
}
