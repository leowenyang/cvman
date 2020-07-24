# -*-coding:utf8-*-#
import pytest

from cvman.cvmod.Subdiv import Subdiv

def test_getTriangleList():
    #Create an instance of Subdiv2d
    rect = (0, 0, 10, 10)
    subdiv = Subdiv(rect)
    subdiv.insertPoint((3, 3))
    assert 4 == len(subdiv.getTriangleList())

def test_getVoronoiFacetList():
    #Create an instance of Subdiv2d
    rect = (0, 0, 10, 10)
    subdiv = Subdiv(rect)
    subdiv.insertPoint((3, 3))
    assert 2 == len(subdiv.getVoronoiFacetList())

if __name__ == "__main__":
    pytest.main()