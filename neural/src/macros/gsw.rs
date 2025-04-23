/*
    Appellation: gsw <module>
    Contrib: @FL03
*/

macro_rules! setter {
    (@impl $name:ident: $T:ty) => {
        paste::item! {
            pub fn [<set_ $name>](&mut self, $name: $T) {
                self.$name = $name;
            }

            pub fn [<with_ $name>] (self, $name: $T) -> Self {
                Self {
                    $name,
                    ..self
                }
            }
        }
    };
    ($($name:ident: $T:ty),* $(,)?) => {
        $(setter!(@impl $name: $T);)*
    };
}

macro_rules! getter {
    (@impl $name:ident: &$T:ty) => {
        pub const fn $name(&self) -> &$T {
            &self.$name
        }
        paste::paste! {
            pub fn [<$name _mut>] (&mut self) -> &mut $T {
                &mut self.$name
            }
        }
    };
    (@impl $name:ident: $T:ty) => {
        pub const fn $name(&self) -> $T {
            self.$name
        }
        paste::paste! {
            pub fn [<$name _mut>] (&mut self) -> &mut $T {
                &mut self.$name
            }
        }
    };
    ($($name:ident: $($rest:tt)*),* $(,)?) => {
        $(getter!(@impl $name: $($rest)*);)*
    };
}

macro_rules! gsw {
    ($($name:ident: &$T:ty),* $(,)?) => {
        $(
            getter!($name: &$T);
            setter!($name: $T);
        )*
    };
    ($($name:ident: $T:ty),* $(,)?) => {
        $(
            getter!($name: $T);
            setter!($name: $T);
        )*
    };

}
