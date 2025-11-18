from sum_of_squares import sum_of_squares

def test_sum_of_squares():
    print("Testing sum_of_squares function...")

    # Test case 1: Small input
    result = sum_of_squares(5)
    expected = 35.0
    print(f"Test 1: sum_of_squares(5) = {result}, Expected: {expected}")
    assert result == expected, f"Test 1 failed. Got {result}, expected {expected}"

    # Test case 2: Another small input
    result = sum_of_squares(7)
    expected = 84.0
    print(f"Test 2: sum_of_squares(7) = {result}, Expected: {expected}")
    assert result == expected, f"Test 2 failed. Got {result}, expected {expected}"

    # Test case 3: Edge case (n = 1)
    result = sum_of_squares(1)
    expected = 1.0
    print(f"Test 3: sum_of_squares(1) = {result}, Expected: {expected}")
    assert result == expected, f"Test 3 failed. Got {result}, expected {expected}"

    # Test case 4: Invalid input (negative number)
    result = sum_of_squares(-5)
    expected = -1
    print(f"Test 4: sum_of_squares(-5) = {result}, Expected: {expected}")
    assert result == expected, f"Test 4 failed. Got {result}, expected {expected}"

    # Test case 5: Invalid input (float)
    result = sum_of_squares(3.14)
    expected = -1
    print(f"Test 5: sum_of_squares(3.14) = {result}, Expected: {expected}")
    assert result == expected, f"Test 5 failed. Got {result}, expected {expected}"

    # Test case 6: Larger nput
    result = sum_of_squares(10)
    expected = 165.0
    print(f"Test 6: sum_of_squares(10) = {result}, Expected: {expected}")
    assert result == expected, f"Test 6 failed. Got {result}, expected {expected}"

    print("All tests passed!")

# Run the tests
test_sum_of_squares()
