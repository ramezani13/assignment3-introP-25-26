from is_odd import *

def test_odd1():
    assert is_odd(3) is True

def test_odd2():
    assert is_odd(6) is False

def test_odd3():
    assert is_odd(0) is False

def test_odd4():
    assert is_odd(-7) is True

def test_restrictions():
    with open('is_odd.py') as f:
        code = f.read()
        if '*' in code:
            raise ValueError("Use of * not permitted in is_odd.py")
        if '/' in code:
            raise ValueError("Use of / not permitted in is_odd.py")
        if '%' in code:
            raise ValueError("Use of % not permitted in is_odd.py")
        

