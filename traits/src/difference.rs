/*
    Appellation: difference <module>
    Created At: 2025.11.26:12:09:51
    Contrib: @FL03
*/

/// Compute the percentage difference between two values.
/// The percentage difference is defined as:
///
/// ```text
/// percent_diff = |x, y| 100 * |x - y| / ((|x| + |y|) / 2)
/// ```
pub trait PercentDiff<Rhs = Self> {
    type Output;

    fn percent_diff(self, rhs: Rhs) -> Self::Output;
}
