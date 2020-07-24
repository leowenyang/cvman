# -*-coding:utf8-*-#
import cvman.cvapp.camera3D as camera
import cvman.cvapp.faceSubdiv as faceSubdiv
import cvman.cvmod.faceDetect as face

def runCamera():
    path = './tests/data/'
    camera.run(path)

def runSubdiv():
    file = './tests/data/face.jpg'
    faceSubdiv.runImg(file, file)

def runFace():
    file = './tests/data/face.jpg'
    face.drawFaces(file)
    face.drawEyes(file)
    face.drawSmiles(file)

def main():
    # runCamera()
    runSubdiv()
    # runFace()


if __name__ == '__main__':
    main()