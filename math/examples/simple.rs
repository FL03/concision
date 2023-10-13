extern crate concision_math as math;

use math::prelude::{complex::C, Numerical};

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
