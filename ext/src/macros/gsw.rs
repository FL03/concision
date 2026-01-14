/*
    appellation: gsw <module>
    authors: @FL03
*/

#[allow(unused_macros)]
/// [`set_with!`] is a macro used to generate `set_` and `with_` methods for a struct's fields.
macro_rules! set_with {
    ($($field:ident: $T:ty),* $(,)?) => {
        $(paste::paste! {
            /// update the current value of the field and return a mutable reference to self
            pub fn [<set_ $field>](&mut self, value: $T) -> &mut Self {
                self.$field = value;
                self
            }
            /// consume the current instance to create another with the given value
            pub fn [<with_ $field>](self, $field: $T) -> Self {
                Self {
                    $field,
                    ..self
                }
            }
        })*
    };
}
