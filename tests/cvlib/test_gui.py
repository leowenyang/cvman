# -*-coding:utf8-*-#
import pytest

import cvman.cvlib.gui as cvgui

def test_imread():
    cvgui.namedWindow("test")
    assert 1 == 0

if __name__ == "__main__":
    pytest.main()