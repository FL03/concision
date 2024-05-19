/*
    Appellation: builder <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_export]
macro_rules! builder {
    ($(#[derive($($d:ident),+)])?$name:ident::<$inner:ty> {$($k:ident: $v:ty),* $(,)?}) => {
        builder!(@loop builder: $name, derive: [$($($d),+)?], inner: $inner {$($k: $v),*});
    };
    ($(#[derive($($d:ident),+)])? $name:ident($inner:ty) {$($k:ident: $v:ty),* $(,)?}) => {
        builder!(@loop builder: $name, derive: [$($($d),+)?], inner: $inner {$($k: $v),*});
    };
    (@loop builder: $name:ident, derive: [$($d:ident),* $(,)?], inner: $inner:ty {$($k:ident: $v:ty),* $(,)?}) => {

        #[derive(Default, $($d),*)]
        pub struct $name {
            inner: $inner,
        }

        builder!(@impl builder: $name, inner: $inner {$($k: $v),*});
    };
    (@impl builder: $name:ident, inner: $inner:ty {$($k:ident: $v:ty),* $(,)?}) => {
        impl $name {
            pub fn new() -> Self {
                Self {
                    inner: Default::default()
                }
            }

            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn build(self) -> $inner {
                self.inner
            }

            $(
                pub fn $k(mut self, $k: $v) -> Self {
                    self.inner.$k = $k;
                    self
                }
            )*
        }
    };
}
