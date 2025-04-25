/*
    Appellation: dataset <module>
    Contrib: @FL03
*/
use super::Records;

#[derive(Clone, Copy, Default, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Dataset<U, V> {
    pub records: U,
    pub targets: V,
}

impl<U, V> Dataset<U, V> {
    pub fn new(records: U, targets: V) -> Self {
        Self { records, targets }
    }

    gsw! {
        records: &U,
        targets: &V,
    }
}

impl<U, V> Records for Dataset<U, V> {
    type Inputs = U;
    type Targets = V;

    fn inputs(&self) -> &Self::Inputs {
        &self.records
    }

    fn inputs_mut(&mut self) -> &mut Self::Inputs {
        &mut self.records
    }

    fn targets(&self) -> &Self::Targets {
        &self.targets
    }

    fn targets_mut(&mut self) -> &mut Self::Targets {
        &mut self.targets
    }
}
