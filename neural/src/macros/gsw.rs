/*
    Appellation: gsw <module>
    Contrib: @FL03
*/

/// The `gsw` macro generates getter and setter methods for the fields of a struct.
///
/// ### Usage
///
/// ```no_run
/// #[derive(Clone, Debug, Default)]
/// pub struct Sample {
///     pub(crate) a: usize,
///     pub(crate) b: std::collections::HashMap<String, usize>,
/// }
///
/// impl Sample {
///    gsw! {
///       a: usize,
///     }
///     gsw! {
///         b: &std::collections::HashMap<String, usize>,
///     }
/// }
///
/// fn _sampler() {
///     let mut sample = Sample::default().with_a(10);
///     assert_eq!(sample.a(), 10);
///     assert_eq!(sample.a_mut(), &mut 10);
///     assert_eq!(sample.set_a(20).a(), 20);
/// }
/// ```
macro_rules! gsw {
    ($($name:ident: $T:ty),* $(,)?) => {
        get_set_with!(@set $($name: $T),*);
        get_set_with!(@with $($name: $T),*);
        get_set_with!(@get $($name: $T),*);
    };
    ($($name:ident: &$T:ty),* $(,)?) => {
        get_set_with!(@set $($name: &$T),*);
        get_set_with!(@with $($name: &$T),*);
        get_set_with!(@get $($name: &$T),*);
    };
}

macro_rules! get_set_with {
    (@get $($name:ident: $T:ty),* $(,)?) => {
        $(
            get_set_with!(@getter $name: $T);
        )*
    };
    (@get $($name:ident: &$T:ty),* $(,)?) => {
        $(
            get_set_with!(@getter $name: &$T);
        )*
    };
    (@set $($name:ident: $(&)?$T:ty),* $(,)?) => {
        $(
            get_set_with!(@setter $name: $T);
        )*
    };
    (@with $($name:ident: $(&)?$T:ty),* $(,)?) => {
        $(
            get_set_with!(@wither $name: $T);
        )*
    };
    (@getter $name:ident: &$T:ty) => {
        pub const fn $name(&self) -> &$T {
            &self.$name
        }
        paste::paste! {
            pub fn [<$name _mut>] (&mut self) -> &mut $T {
                &mut self.$name
            }
        }
    };
    (@getter $name:ident: $T:ty) => {
        pub const fn $name(&self) -> $T {
            self.$name
        }
        paste::paste! {
            pub fn [<$name _mut>] (&mut self) -> &mut $T {
                &mut self.$name
            }
        }
    };
    (@setter $name:ident: $T:ty) => {
        paste::item! {
            pub fn [<set_ $name>](&mut self, $name: $T) -> &mut Self {
                self.$name = $name;
                self
            }
        }
    };
    (@wither $name:ident: $T:ty) => {
        paste::item! {
            pub fn [<with_ $name>] (self, $name: $T) -> Self {
                Self {
                    $name,
                    ..self
                }
            }
        }
    };
}
