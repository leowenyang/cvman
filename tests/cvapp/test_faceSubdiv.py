# -*-coding:utf8-*-#
import os
import pytest
import tempfile

import cvman.cvapp.faceSubdiv as faceSubdiv

def test_runImg():
    outFile = os.path.join(tempfile.gettempdir(), "face1.jpg")
    faceSubdiv.runImg("./tests/data/face.jpg", outFile)
    assert os.path.isfile(outFile)

def test_runVideo():
    outFile = os.path.join(tempfile.gettempdir(), "face.mp4")
    faceSubdiv.runVideo("./tests/data/face.jpg", outFile)
    assert os.path.isfile(outFile)

if __name__ == "__main__":
    pytest.main()