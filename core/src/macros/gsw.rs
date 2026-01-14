/*
    appellation: gsw <module>
    authors: @FL03
*/
#![cfg(feature = "macros")]

#[macro_export]
/// [`get!`] is a macro used to generate get, set, and with methods for a struct's fields.
macro_rules! get {
    ($($field:ident: $T:ty),* $(,)?) => {
        paste::paste! {
            $(
                /// returns an immutable reference to the field
                pub const fn $field(&self) -> $T {
                    self.$field
                }
                /// return a mutable reference to the field
                pub const fn [<$field _mut>](&mut self) -> &mut $T {
                    &mut self.$field
                }
            )*
        }
    };
    ($($field:ident: &$T:ty),* $(,)?) => {
        paste::paste! {
            $(
                /// returns an immutable reference to the field
                pub const fn $field(&self) -> &$T {
                    &self.$field
                }
                /// return a mutable reference to the field
                pub const fn [<$field _mut>](&mut self) -> &mut $T {
                    &mut self.$field
                }
            )*
        }
    };
}
#[macro_export]
/// [`set_with!`] is a macro used to generate `set_` and `with_` methods for a struct's fields.
macro_rules! set_with {
    ($($field:ident: $F:ty),* $(,)?) => {
        paste::paste! {
            $(
                /// update the current value of the field and return a mutable reference to self
                pub fn [<set_ $field>](&mut self, value: $T) {
                    self.$field = value
                }
                /// consume the current instance to create another with the given value
                pub fn [<with_ $field>](self, $field: $T) -> Self {
                    Self {
                        $field,
                        ..self
                    }
                }
            )*
        }

    };
}
