pub trait PercentDiff<Rhs = Self> {
    type Output;

    fn percent_diff(self, rhs: Rhs) -> Self::Output;
}
