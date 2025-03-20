# Project: Number Guessing Game

import random

def number_guessing_game():
    """
    A simple number guessing game where the user tries to guess
    a randomly generated number between 1 and 100.
    """
    print("Welcome to the Number Guessing Game! 🎮")
    print("I'm thinking of a number between 1 and 100...")
    
    # Step 1: Generate a random number
    number_to_guess = random.randint(1, 100)
    attempts = 0
    max_attempts = 10  # For bonus task
    
    # Step 2: Prompt the user for guesses
    while attempts < max_attempts:
        try:
            # Get user's guess
            guess = int(input("\nGuess the number (between 1 and 100): "))
            attempts += 1
            
            # Check if the guess is correct
            if guess == number_to_guess:
                print(f"Congratulations! You guessed it in {attempts} attempts! 🎉")
                break
            # Give hints
            elif guess > number_to_guess:
                print("Too high! Try again.")
            else:
                print("Too low! Try again.")
            
            # Show remaining attempts
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"You have {remaining} attempts left.")
            
        except ValueError:
            print("Please enter a valid number.")
    
    # Check if user ran out of attempts
    if attempts == max_attempts and guess != number_to_guess:
        print(f"\nGame over! Better luck next time! 😢")
        print(f"The number was {number_to_guess}.")
    
    # Ask if the user wants to play again
    play_again = input("\nWould you like to play again? (yes/no): ").lower()
    if play_again.startswith('y'):
        number_guessing_game()  # Restart the game
    else:
        print("Thanks for playing! Goodbye! 👋")

# Start the game
if __name__ == "__main__":
    number_guessing_game()
