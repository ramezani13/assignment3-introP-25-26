from is_palindrome import *

def test_palindrome1():
    assert is_palindrome(-4) is False

def test_palindrome2():
    assert is_palindrome(3) is True

def test_palindrome3():
    assert is_palindrome(20572) is False

def test_palindrome4():
    assert is_palindrome(1234321) is True

def test_palindrome5():
    assert is_palindrome(12344321) is True

