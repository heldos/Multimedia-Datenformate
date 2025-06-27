{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.conda
    pkgs.python312Packages.conda-libmamba-solver
    pkgs.zlib
    pkgs.libjpeg
    pkgs.libpng
    pkgs.ffmpeg
    pkgs.pkg-config
    pkgs.stdenv.cc.cc.lib   # libstdc++.so.6
    pkgs.mesa               # OpenGL support
    pkgs.libGL
    pkgs.libGLU
    pkgs.xorg.libX11
    pkgs.glib               # <- this provides libgthread-2.0.so.0
    pkgs.libmamba
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.python312Packages.conda-libmamba-solver}/lib:${pkgs.zlib}/lib:{pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.mesa}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.xorg.libX11}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
    export VIRTUAL_ENV=.venv
    [ ! -d $VIRTUAL_ENV ] && python3 -m venv $VIRTUAL_ENV
    source $VIRTUAL_ENV/bin/activate
  '';
}

# kein manuell
# biom metrics like erm
# vergleich normale (wie ssm) mit biom (erm)
# learning based comrpresion