#[cfg(test)]

#[test]
fn lib_compiles() {
    let f = | x: usize, y: usize | x + y;

    assert_eq!(f(10, 10), 20)
}