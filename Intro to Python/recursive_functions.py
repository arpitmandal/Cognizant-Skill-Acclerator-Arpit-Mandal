# Project: Menu of Recursive Functions

import time
try:
    import turtle  # For the bonus fractal pattern
    turtle_available = True
except ImportError:
    turtle_available = False

def factorial(n):
    """Calculate the factorial of a number using recursion."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 0:
        return "Input should be a positive integer"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def draw_fractal_tree(t, branch_length, angle, level):
    """Draw a recursive fractal tree using the turtle module."""
    if level == 0:
        return
    
    # Draw the current branch
    t.forward(branch_length)
    
    # Draw right branch
    t.right(angle)
    draw_fractal_tree(t, branch_length * 0.7, angle, level - 1)
    
    # Draw left branch
    t.left(angle * 2)
    draw_fractal_tree(t, branch_length * 0.7, angle, level - 1)
    
    # Return to the starting position of this branch
    t.right(angle)
    t.backward(branch_length)

def get_positive_int(prompt):
    """Get a positive integer from the user with validation."""
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                print("Please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main program
print("Welcome to the Recursive Artistry Program!")

while True:
    print("\nChoose an option:")
    print("1. Calculate Factorial")
    print("2. Find Fibonacci")
    if turtle_available:
        print("3. Draw a Recursive Fractal")
    print("4. Exit")
    
    choice = input("> ")
    
    if choice == "1":
        # Factorial calculation
        n = get_positive_int("Enter a number to find its factorial: ")
        if n > 20:
            print("Warning: Large values may cause a long calculation time.")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                continue
        
        result = factorial(n)
        print(f"The factorial of {n} is {result}.")
    
    elif choice == "2":
        # Fibonacci calculation
        n = get_positive_int("Enter the position of the Fibonacci number: ")
        if n > 30:
            print("Warning: Due to the recursive implementation, values > 30 may take a long time.")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                continue
        
        result = fibonacci(n)
        print(f"The {n}th Fibonacci number is {result}.")
    
    elif choice == "3" and turtle_available:
        # Draw fractal tree
        print("Drawing a recursive fractal tree...")
        level = min(get_positive_int("Enter the recursion depth (1-10 recommended): "), 10)
        
        # Set up turtle
        t = turtle.Turtle()
        wn = turtle.Screen()
        wn.bgcolor("black")
        t.color("green")
        t.speed(0)  # Fastest speed
        
        # Position turtle
        t.left(90)
        t.up()
        t.backward(300)
        t.down()
        
        # Draw the tree
        draw_fractal_tree(t, 120, 30, level)
        
        # Wait for click to continue
        print("Click on the turtle window to continue...")
        wn.exitonclick()
    
    elif choice == "4":
        print("Thank you for using the Recursive Artistry Program. Goodbye!")
        break
    
    else:
        print("Invalid choice. Please enter a valid option.")
