from count_odd_digits import *

def test_count1():
    assert count_odd_digits(123) == 2

def test_count2():
    assert count_odd_digits(2468) == 0

def test_count3():
    assert count_odd_digits(13579) == 5

def test_count4():
    assert count_odd_digits(0) == 0

def test_count5():
    assert count_odd_digits(-123) == -1

def test_restrictions():
    with open('count_odd_digits.py') as f:
        code = f.read()
    if '*' in code:
        raise ValueError("Use of * not permitted in count_odd_digits.py")
    if '/' in code and '//' not in code:
        raise ValueError("Use of / not permitted in count_odd_digits.py")
    if '%' in code:
        raise ValueError("Use of % not permitted in count_odd_digits.py")
    if 'str' in code:
        raise ValueError("Converting to string not permitted in count_odd_digits.py")
