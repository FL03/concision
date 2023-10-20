#[cfg(test)]
#[test]
fn compiles() {
    let f = |x: usize, y: usize| x + y;

    assert_eq!(f(10, 10), 20);
    assert_ne!(f(1, 1), 3);
}
