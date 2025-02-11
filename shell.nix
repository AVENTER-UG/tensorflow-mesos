with import <nixpkgs> {};

stdenv.mkDerivation {
name = "python-env";

buildInputs = [
    python311Full
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.tkinter
    lighttpd
];

SOURCE_DATE_EPOCH = 315532800;
PROJDIR = "/tmp/python-dev";
S_NETWORK="host";

shellHook = ''
    echo "Using ${python311.name}"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib"
    
    [ ! -d '$PROJDIR' ] && virtualenv $PROJDIR && echo "SETUP python-dev: DONE"
    source $PROJDIR/bin/activate
    mkdir -p /mnt/mesos/sandbox/
    tar -xvf examples/download/flower_photos.tgz -C /mnt/mesos/sandbox/

    pip install avmesos
    pip install matplotlib
    pip install ipykernel
    pip install tensorflow==2.13.1

    make install-dev

    '';
}
