/*
    Appellation: default <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[test]
fn lib_compiles() {
    fn add<A, B, C>(a: A, b: B) -> C
    where
        A: core::ops::Add<B, Output = C>,
    {
        a + b
    }
    assert_eq!(add(10, 10), 20);
    assert_ne!(add(1, 1), 3);
}
