# Project: Calculator with Exception Handling

import logging

# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                   format='%(asctime)s:%(levelname)s:%(message)s')

def get_number(prompt):
    """Get a valid number from the user with exception handling."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a valid number.")

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b with exception handling for division by zero."""
    try:
        return a / b
    except ZeroDivisionError:
        logging.error("ZeroDivisionError occurred: division by zero.")
        print("Oops! Division by zero is not allowed.")
        return None

# Main program
print("Welcome to the Error-Free Calculator!")

while True:
    # Display menu
    print("\nChoose an operation:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")
    
    try:
        # Get user choice
        choice = int(input("> "))
        
        # Exit condition
        if choice == 5:
            print("Goodbye!")
            break
        
        # Validate choice
        if choice < 1 or choice > 5:
            print("Please select a valid option (1-5).")
            continue
        
        # Get input numbers
        num1 = get_number("Enter the first number: ")
        num2 = get_number("Enter the second number: ")
        
        # Perform the selected operation
        if choice == 1:
            result = add(num1, num2)
            print(f"{num1} + {num2} = {result}")
        elif choice == 2:
            result = subtract(num1, num2)
            print(f"{num1} - {num2} = {result}")
        elif choice == 3:
            result = multiply(num1, num2)
            print(f"{num1} × {num2} = {result}")
        elif choice == 4:
            result = divide(num1, num2)
            if result is not None:
                print(f"{num1} ÷ {num2} = {result}")
        
    except ValueError:
        print("Please enter a valid number for the menu option.")
        logging.error("ValueError: Invalid menu selection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {str(e)}")
