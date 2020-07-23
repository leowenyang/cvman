# -*-coding:utf8-*-#
import pytest

import cvman.cvmod.faceDetect as faceDetect

def test_detectFacesByDlib():
    faces = faceDetect.detectFacesByDlib("./tests/data/face.jpg")
    assert 1 == len(faces)

def test_detectLandMarkByDlib():
    faces = faceDetect.detectLandMarkByDlib("./tests/data/face.jpg")
    assert 68 == len(faces)

def test_detectFaces():
    faces = faceDetect.detectFaces("./tests/data/face.jpg")
    assert 1 == len(faces)

def test_drawFaces():
    # cvgui.namedWindow("test")
    assert 1 == 1

def test_detectEyes():
    eyes = faceDetect.detectEyes("./tests/data/face.jpg")
    assert 3 == len(eyes)

def test_drawEyes():
    # cvgui.namedWindow("test")
    assert 1 == 1

def test_detectSmiles():
    smiles = faceDetect.detectSmiles("./tests/data/face.jpg")
    assert 2 == len(smiles)

def test_drawSmiles():
    # cvgui.namedWindow("test")
    assert 1 == 1




if __name__ == "__main__":
    pytest.main()