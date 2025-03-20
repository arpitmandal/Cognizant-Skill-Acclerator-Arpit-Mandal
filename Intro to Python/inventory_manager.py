# Project: Inventory Management

def display_inventory(inventory):
    """Display all items in the inventory with their quantities and prices."""
    print("\nCurrent inventory:")
    for item, (quantity, price) in inventory.items():
        print(f"Item: {item}, Quantity: {quantity}, Price: ${price}")

def calculate_total_value(inventory):
    """Calculate the total value of the inventory."""
    total = 0
    for item, (quantity, price) in inventory.items():
        total += quantity * price
    return total

# Main program
print("Welcome to the Inventory Manager!")

# Initialize an empty inventory
inventory = {}

# Add some initial items
inventory["apple"] = (10, 2.5)
inventory["banana"] = (20, 1.2)

# Display initial inventory
display_inventory(inventory)

# Menu loop
while True:
    print("\nPlease select an option:")
    print("1. Add a new item")
    print("2. Remove an item")
    print("3. Update an item")
    print("4. Display inventory")
    print("5. Calculate total value")
    print("6. Exit")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == "1":
        # Add a new item
        item_name = input("Enter the name of the new item: ")
        
        if item_name in inventory:
            print(f"Item '{item_name}' already exists in the inventory.")
            continue
        
        try:
            quantity = int(input("Enter the quantity: "))
            price = float(input("Enter the price: $"))
            
            if quantity <= 0 or price <= 0:
                print("Quantity and price must be positive values.")
                continue
            
            inventory[item_name] = (quantity, price)
            print(f"Added: {item_name}")
        except ValueError:
            print("Please enter valid numbers for quantity and price.")
    
    elif choice == "2":
        # Remove an item
        item_name = input("Enter the name of the item to remove: ")
        
        if item_name in inventory:
            del inventory[item_name]
            print(f"Removed: {item_name}")
        else:
            print(f"Item '{item_name}' not found in the inventory.")
    
    elif choice == "3":
        # Update an item
        item_name = input("Enter the name of the item to update: ")
        
        if item_name in inventory:
            update_choice = input("Update (q)uantity or (p)rice? ").lower()
            
            if update_choice == 'q':
                try:
                    new_quantity = int(input("Enter the new quantity: "))
                    if new_quantity <= 0:
                        print("Quantity must be a positive value.")
                        continue
                    
                    current_price = inventory[item_name][1]
                    inventory[item_name] = (new_quantity, current_price)
                    print(f"Updated quantity for {item_name}.")
                except ValueError:
                    print("Please enter a valid number for quantity.")
            
            elif update_choice == 'p':
                try:
                    new_price = float(input("Enter the new price: $"))
                    if new_price <= 0:
                        print("Price must be a positive value.")
                        continue
                    
                    current_quantity = inventory[item_name][0]
                    inventory[item_name] = (current_quantity, new_price)
                    print(f"Updated price for {item_name}.")
                except ValueError:
                    print("Please enter a valid number for price.")
            
            else:
                print("Invalid choice. Please enter 'q' for quantity or 'p' for price.")
        else:
            print(f"Item '{item_name}' not found in the inventory.")
    
    elif choice == "4":
        # Display inventory
        display_inventory(inventory)
    
    elif choice == "5":
        # Calculate total value
        total_value = calculate_total_value(inventory)
        print(f"\nTotal value of inventory: ${total_value:.2f}")
    
    elif choice == "6":
        # Exit the program
        print("Thank you for using the Inventory Manager. Goodbye!")
        break
    
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")
