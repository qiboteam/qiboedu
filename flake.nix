{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv = {
      url = "github:cachix/devenv";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs-python = {
      url = "github:cachix/nixpkgs-python";
      inputs = {nixpkgs.follows = "nixpkgs";};
    };
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;

          modules = [
            ({
              lib,
              pkgs,
              config,
              ...
            }: {
              packages = with pkgs; [pre-commit poethepoet];

              env = {
                LD_LIBRARY_PATH = builtins.concatStringsSep ":" (map (p: "${p}/lib") (with pkgs; [
                  stdenv.cc.cc.lib
                  zlib
                ]));
                PYTHONBREAKPOINT = "pudb.set_trace";
              };

              enterShell = ''
                export PATH="$DEVENV_ROOT/.devenv/state/venv/bin:$PATH"
              '';

              languages.python = {
                enable = true;
                libraries = with pkgs; [zlib];
                venv = {
                  enable = true;
                  requirements = ''
                    numpy
                    scipy
                    matplotlib

                    # Qibo
                    qibo
                    qibolab[emulator]
                    qibocal

                    # notebooks
                    jupyterlab
                    ipython

                    # dev deps
                    pudb
                  '';
                };
              };
            })
          ];
        };
      });
  };
}
