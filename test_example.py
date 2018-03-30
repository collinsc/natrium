import unittest
import pytest


def test_pass():
    assert True


# def test_fail():
#     assert False


def test_exception():
    with pytest.raises(IOError):
        with open('nonexistent_file.txt') as f:
            print(f)


if __name__ == '__main__':
    unittest.main()
