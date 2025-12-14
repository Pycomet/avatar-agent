# Mock menu data for restaurant assistant
# This can be easily replaced with an API call to fetch real menu data

MENU_ITEMS = [
    {
        "id": "1",
        "name": "Bruschetta",
        "price": 8.99,
        "category": "Appetizers",
        "description": "Fresh tomatoes, basil, and mozzarella on toasted bread",
        "image_url": "https://images.unsplash.com/photo-1572695157366-5e585ab2b69f?w=400",
    },
    {
        "id": "2",
        "name": "Pasta Carbonara",
        "price": 14.99,
        "category": "Mains",
        "description": "Classic Roman pasta with eggs, cheese, and pancetta",
        "image_url": "https://images.unsplash.com/photo-1612874742237-6526221588e3?w=400",
    },
    {
        "id": "3",
        "name": "Margherita Pizza",
        "price": 12.99,
        "category": "Mains",
        "description": "Traditional pizza with tomato, mozzarella, and basil",
        "image_url": "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=400",
    },
    {
        "id": "4",
        "name": "Caesar Salad",
        "price": 9.99,
        "category": "Appetizers",
        "description": "Romaine lettuce with parmesan, croutons, and Caesar dressing",
        "image_url": "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400",
    },
    {
        "id": "5",
        "name": "Tiramisu",
        "price": 7.99,
        "category": "Desserts",
        "description": "Classic Italian dessert with coffee-soaked ladyfingers",
        "image_url": "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=400",
    },
    {
        "id": "6",
        "name": "Espresso",
        "price": 3.99,
        "category": "Drinks",
        "description": "Rich Italian espresso",
        "image_url": "https://images.unsplash.com/photo-1510591509098-f4fdc6d0ff04?w=400",
    },
]


def get_all_items():
    """Get all menu items."""
    return MENU_ITEMS


def get_item_by_id(item_id: str):
    """Get a specific menu item by ID."""
    for item in MENU_ITEMS:
        if item["id"] == item_id:
            return item
    return None


def find_item_by_name(name: str):
    """Find a menu item by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    for item in MENU_ITEMS:
        if name_lower in item["name"].lower():
            return item
    return None

