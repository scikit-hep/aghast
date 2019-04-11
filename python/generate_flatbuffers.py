#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import os
import shutil

if __name__ == "__main__":
    if os.path.exists(os.path.join("aghast", "aghast_generated")):
        shutil.rmtree(os.path.join("aghast", "aghast_generated"))

    os.chdir("aghast")
    os.system("flatc --python ../../flatbuffers/aghast.fbs")

    with open(os.path.join("aghast_generated", "StatisticFilter.py")) as f:
        tmp = f.read().replace("inf.0", "float('inf')")
    with open(os.path.join("aghast_generated", "StatisticFilter.py"), "w") as f:
        f.write(tmp)
