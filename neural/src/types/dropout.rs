/*
    Appellation: dropout <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

/// The [Dropout] layer is randomly zeroizes inputs with a given probability (`p`).
/// This regularization technique is often used to prevent overfitting.
///
///
/// ### Config
///
/// - (p) Probability of dropping an element
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Dropout {
    pub(crate) p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(feature = "rand")]
impl<U> cnc::Forward<U> for Dropout
where
    U: cnc::DropOut,
{
    type Output = <U as cnc::DropOut>::Output;

    fn forward(&self, input: &U) -> Option<Self::Output> {
        Some(input.dropout(self.p))
    }
}
