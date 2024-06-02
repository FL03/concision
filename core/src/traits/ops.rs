/*
    Appellation: ops <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
/// A trait for applying a function to a type
pub trait Apply<T, F> {
    type Output;

    fn apply<U>(&self, f: F) -> Self::Output
    where
        F: Fn(T) -> U;
}

/// An alternative trait for evaluating the logic of an expression;
/// [Eval] is a substitute for functional traits (Fn, FnMut, and FnOnce) as
/// implementing these traits is currently unstable.
pub trait Eval<T> {
    type Output;

    fn eval(&self, args: T) -> Self::Output;
}
/// [EvaluateLazy] is used for _lazy_, structured functions that evaluate to
/// some value.
pub trait EvaluateLazy {
    type Output;

    fn eval(&self) -> Self::Output;
}

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<X, Y, F> Eval<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn eval(&self, args: X) -> Self::Output {
        self(args)
    }
}

impl<X, Y> Eval<X> for Box<dyn Eval<X, Output = Y>> {
    type Output = Y;

    fn eval(&self, args: X) -> Self::Output {
        self.as_ref().eval(args)
    }
}
