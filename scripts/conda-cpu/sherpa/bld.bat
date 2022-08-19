

python setup.py install --single-version-externally-managed --record=record.txt

COPY build/lib.win-amd64-*/sherpa/bin/sherpa.exe %LIBRARY_BIN% || exit 1
COPY build/lib.win-amd64-*/sherpa/lib/sherpa_core.lib %LIBRARY_LIB% || exit 1
