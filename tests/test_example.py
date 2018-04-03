"""
A selection of example tests to show how to use basic pylint
"""
import unittest
import pytest


def test_pass():
    """
    Sample of a passing test
    :return:
    """
    assert True


# def test_fail():
#     assert False


def test_exception():
    """
    Sample of how to check for a raised exception
    :return:
    """
    with pytest.raises(IOError):
        with open('nonexistent_file.txt') as file:
            print(file)


if __name__ == '__main__':
    unittest.main()
