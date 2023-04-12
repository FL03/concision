/*
    Appellation: num <module>
    Contrib: FL03 <jo3mccain@icloud.com>
    Description: ... Summary ...
*/

pub fn central<T: Copy + num::Num>(x: T, f: Box<dyn Fn(T) -> T>, h: T) -> T {
    (f(x + h) - f(x)) / h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central() {
        let f = |x: f64| x * x;
        let res = central(5.0, Box::new(f), 0.001);
        assert_eq!(res.round(), 10.0);
    }
}
