/*
    Appellation: dataset <module>
    Contrib: @FL03
*/
use super::Records;

/// A dataset is a collection of records and targets along with various other attributes useful
/// for machine learning tasks
#[derive(Clone, Copy, Default, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct DatasetBase<U, V> {
    pub records: U,
    pub targets: V,
}

impl<U, V> DatasetBase<U, V> {
    pub fn new(records: U, targets: V) -> Self {
        Self { records, targets }
    }

    pub const fn records(&self) -> &U {
        &self.records
    }
    #[inline]
    pub const fn records_mut(&mut self) -> &mut U {
        &mut self.records
    }

    pub const fn targets(&self) -> &V {
        &self.targets
    }
    #[inline]
    pub const fn targets_mut(&mut self) -> &mut V {
        &mut self.targets
    }
}

impl<U, V> core::fmt::Display for DatasetBase<U, V>
where
    U: core::fmt::Display,
    V: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{\n  records: {},\n  targets: {}\n}}",
            self.records, self.targets
        )
    }
}

impl<U, V> Records for DatasetBase<U, V> {
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
