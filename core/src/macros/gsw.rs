/*
    appellation: gsw <module>
    authors: @FL03
*/
#![cfg(feature = "macros")]

#[macro_export]
/// [`gsw!`] is a macro used to generate get, set, and with methods for a struct's fields.
macro_rules! gsw {
    ($name:ident$(<$T:ident>)? {$($field:ident: $F:ty),* $(,)?}
    ) => {

        impl$(<$T>)? $name$(<$T>)? {
            gsw!(@get $($field: $F),*);
            gsw!(@set $($field: $F),*);
            gsw!(@with $($field: $F),*);
        }

    };
    (@get $($field:ident: $T:ty),* $(,)?) => {
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
    (@set $($field:ident: $T:ty),* $(,)?) => {
        paste::paste! {
            $(
                /// update the current value of the field and return a mutable reference to self
                pub fn [<set_ $field>](&mut self, value: $T) -> &mut Self {
                    self.$field = value;
                    self
                }
            )*
        }
    };
    (@with $($field:ident: $T:ty),* $(,)?) => {
        paste::paste! {
            $(
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
