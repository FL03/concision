{
  description = "A flake for the scsys ecosystem.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        rustPlatform = pkgs.rustPlatform;
      in
      {
        packages.default = rustPlatform.buildRustPackage {
          pname = "concision";
          version = "0.0.3";
          src = "./.";
          # If Cargo.lock doesn't exist yet, remove or comment out this block:
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          doCheck = true;
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.cargo-binstall
            pkgs.cargo-watch
            pkgs.cargo-nextest
            pkgs.clippy
            pkgs.rustfmt
          ];
          shellHook = ''
            echo "Welcome to the dev shell!"
            # Add any additional environment setup here
          '';
        };
      }
    );
}