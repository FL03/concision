pub trait FloorDiv<Rhs = Self> {
    type Output;

    fn floor_div(self, rhs: Rhs) -> Self::Output;
}

pub trait RoundTo {
    fn round_to(&self, places: usize) -> Self;
}

impl<T> FloorDiv for T
where
    T: Copy + num_traits::Num,
{
    type Output = T;

    fn floor_div(self, rhs: Self) -> Self::Output {
        crate::floor_div(self, rhs)
    }
}

impl<T> RoundTo for T
where
    T: num_traits::Float,
{
    fn round_to(&self, places: usize) -> Self {
        crate::round_to(*self, places)
    }
}
