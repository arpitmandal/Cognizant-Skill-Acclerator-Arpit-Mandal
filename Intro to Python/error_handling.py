# Assignment: Check your Knowledge on Errors

import logging

# Set up logging for Task 4 (Bonus)
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                   format='%(asctime)s:%(levelname)s:%(message)s')

# Task 1 - Understanding Python Exceptions
print("\n--- Task 1: Understanding Python Exceptions ---")

def divide_by_number():
    """Function to divide 100 by a user-input number."""
    try:
        number = float(input("Enter a number: "))
        result = 100 / number
        print(f"100 divided by {number} is {result}.")
    except ZeroDivisionError:
        print("Oops! You cannot divide by zero.")
    except ValueError:
        print("Invalid input! Please enter a valid number.")

# Run Task 1
divide_by_number()

# Task 2 - Types of Exceptions
print("\n--- Task 2: Types of Exceptions ---")

def demonstrate_exceptions():
    """Function to demonstrate various types of exceptions."""
    # IndexError demonstration
    try:
        my_list = [1, 2, 3]
        # Trying to access an index that doesn't exist
        print(my_list[10])
    except IndexError:
        print("IndexError occurred! List index out of range.")
        # This error occurs when trying to access an index that is out of range

    # KeyError demonstration
    try:
        my_dict = {"name": "Alice", "age": 25}
        # Trying to access a key that doesn't exist
        print(my_dict["address"])
    except KeyError:
        print("KeyError occurred! Key not found in the dictionary.")
        # This error occurs when trying to access a non-existent key in a dictionary

    # TypeError demonstration
    try:
        # Trying to add a string and an integer
        result = "5" + 5
    except TypeError:
        print("TypeError occurred! Unsupported operand types.")
        # This error occurs when performing an operation on incompatible types

# Run Task 2
demonstrate_exceptions()

# Task 3 - Using try...except...else...finally
print("\n--- Task 3: Using try...except...else...finally ---")

def safe_division():
    """Function to safely divide two numbers with comprehensive error handling."""
    try:
        # Try block - attempt the division
        first_number = float(input("Enter the first number: "))
        second_number = float(input("Enter the second number: "))
        result = first_number / second_number
    except ValueError:
        # Handle invalid input
        print("Error: Please enter valid numbers.")
    except ZeroDivisionError:
        # Handle division by zero
        print("Error: Cannot divide by zero.")
    else:
        # Execute if no exceptions occur
        print(f"The result is {result}.")
    finally:
        # Always executes, regardless of exceptions
        print("This block always executes.")

# Run Task 3
safe_division()
