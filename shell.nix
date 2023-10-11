with import <nixpkgs> {};

stdenv.mkDerivation {
name = "python-env";

buildInputs = [
    python310
    python310Packages.pip
    python310Packages.virtualenv
];

SOURCE_DATE_EPOCH = 315532800;
PROJDIR = "${toString ./.}";

shellHook = ''
    echo "Using ${python310.name}"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib"
    
    [ ! -d '$PROJDIR/python-dev' ] && virtualenv python-dev && echo "SETUP python-dev: DONE"
    source python-dev/bin/activate
    pip install avmesos
    make install-dev

    '';
}
