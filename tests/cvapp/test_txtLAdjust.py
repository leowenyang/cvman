# -*-coding:utf8-*-#
import os
import pytest
import tempfile

import cvman.cvapp.txtLAdjust as txtLAdjust

def test_run():
    outFile = os.path.join(tempfile.gettempdir(), "txt1.png")
    txtLAdjust.run("./tests/data/txt.png", outFile)
    assert os.path.isfile(outFile)

if __name__ == "__main__":
    pytest.main()