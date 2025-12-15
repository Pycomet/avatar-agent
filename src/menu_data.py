# Menu data service - fetches from API and provides helper functions
# Supports nested multi-restaurant menu structure:
# restaurants[] -> categories[] -> items[]

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("menu_data")

# Default API URL (can be overridden via environment variable)
MENU_API_URL = os.getenv("MENU_API_URL", "http://localhost:3000/api/menu")


class MenuData:
    """Holds menu data fetched from the API with helper methods.

    Expected API structure:
    {
      "restaurants": [
        {
          "id": "bella-italia",
          "name": "Bella Italia",
          "cuisine": "Italian",
          "image": "...",
          "categories": [
            {
              "id": "pasta",
              "name": "Pasta",
              "items": [
                {
                  "id": "spaghetti-carbonara",
                  "name": "Spaghetti Carbonara",
                  "description": "...",
                  "price": 18.99,
                  "image": "..."
                }
              ]
            }
          ]
        }
      ]
    }
    """

    def __init__(self, data: dict):
        self.raw_data = data
        self.restaurants = data.get("restaurants", [])

    def get_all_restaurants(self) -> list:
        """Get all available restaurants (without nested categories/items for brevity)."""
        return [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "cuisine": r.get("cuisine", ""),
                "image": r.get("image", ""),
            }
            for r in self.restaurants
        ]

    def get_restaurant_by_id(self, restaurant_id: str) -> Optional[dict]:
        """Get a restaurant by ID (full data including categories and items)."""
        for restaurant in self.restaurants:
            if str(restaurant.get("id")) == str(restaurant_id):
                return restaurant
        return None

    def find_restaurant_by_name(self, name: str) -> Optional[dict]:
        """Find a restaurant by name (case-insensitive, partial match)."""
        name_lower = name.lower()
        for restaurant in self.restaurants:
            if name_lower in restaurant.get("name", "").lower():
                return restaurant
        return None

    def get_categories_for_restaurant(self, restaurant_id: str) -> list:
        """Get all categories for a specific restaurant."""
        restaurant = self.get_restaurant_by_id(restaurant_id)
        if not restaurant:
            return []
        return restaurant.get("categories", [])

    def get_items_for_restaurant(self, restaurant_id: str) -> list:
        """Get all menu items for a specific restaurant (flattened from categories)."""
        restaurant = self.get_restaurant_by_id(restaurant_id)
        if not restaurant:
            return []

        items = []
        for category in restaurant.get("categories", []):
            category_name = category.get("name", "")
            for item in category.get("items", []):
                # Add category info to each item for context
                item_with_category = {**item, "categoryName": category_name}
                items.append(item_with_category)
        return items

    def get_items_for_category(self, restaurant_id: str, category_id: str) -> list:
        """Get all menu items in a specific category."""
        restaurant = self.get_restaurant_by_id(restaurant_id)
        if not restaurant:
            return []

        for category in restaurant.get("categories", []):
            if str(category.get("id")) == str(category_id):
                return category.get("items", [])
        return []

    def get_items_by_category_name(
        self, restaurant_id: str, category_name: str
    ) -> list:
        """Get items filtered by category name within a restaurant."""
        restaurant = self.get_restaurant_by_id(restaurant_id)
        if not restaurant:
            return []

        category_name_lower = category_name.lower()
        for category in restaurant.get("categories", []):
            if category_name_lower in category.get("name", "").lower():
                items = category.get("items", [])
                # Add category name to items
                return [
                    {**item, "categoryName": category.get("name")} for item in items
                ]
        return []

    def get_item_by_id(
        self, item_id: str, restaurant_id: Optional[str] = None
    ) -> Optional[dict]:
        """Get a specific menu item by ID, optionally within a restaurant."""
        restaurants_to_search = self.restaurants
        if restaurant_id:
            restaurant = self.get_restaurant_by_id(restaurant_id)
            restaurants_to_search = [restaurant] if restaurant else []

        for restaurant in restaurants_to_search:
            for category in restaurant.get("categories", []):
                for item in category.get("items", []):
                    if str(item.get("id")) == str(item_id):
                        return {
                            **item,
                            "categoryName": category.get("name"),
                            "restaurantId": restaurant.get("id"),
                            "restaurantName": restaurant.get("name"),
                        }
        return None

    def find_item_by_name(
        self, name: str, restaurant_id: Optional[str] = None
    ) -> Optional[dict]:
        """Find a menu item by name (case-insensitive, partial match)."""
        name_lower = name.lower()

        restaurants_to_search = self.restaurants
        if restaurant_id:
            restaurant = self.get_restaurant_by_id(restaurant_id)
            restaurants_to_search = [restaurant] if restaurant else []

        for restaurant in restaurants_to_search:
            for category in restaurant.get("categories", []):
                for item in category.get("items", []):
                    if name_lower in item.get("name", "").lower():
                        return {
                            **item,
                            "categoryName": category.get("name"),
                            "restaurantId": restaurant.get("id"),
                            "restaurantName": restaurant.get("name"),
                        }
        return None

    def get_restaurant_summary(self) -> str:
        """Get a brief summary of available restaurants for the agent."""
        if not self.restaurants:
            return "No restaurants available."

        summaries = []
        for r in self.restaurants:
            name = r.get("name", "Unknown")
            cuisine = r.get("cuisine", "")
            summary = f"- {name}"
            if cuisine:
                summary += f" ({cuisine})"
            summaries.append(summary)

        return "\n".join(summaries)


async def fetch_menu_data() -> MenuData:
    """Fetch menu data from the API endpoint."""
    logger.info(f"Fetching menu data from {MENU_API_URL}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(MENU_API_URL, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            # Count items for logging
            restaurants = data.get("restaurants", [])
            total_items = sum(
                len(item)
                for r in restaurants
                for cat in r.get("categories", [])
                for item in [cat.get("items", [])]
            )

            logger.info(
                f"Menu data fetched: {len(restaurants)} restaurants, "
                f"{total_items} total items"
            )
            return MenuData(data)
    except httpx.RequestError as e:
        logger.error(f"Failed to fetch menu data: {e}")
        # Return empty menu data on error
        return MenuData({"restaurants": []})
    except Exception as e:
        logger.error(f"Unexpected error fetching menu data: {e}")
        return MenuData({"restaurants": []})


# Legacy compatibility functions (for existing code that might use them)
# These work with a global menu instance that must be set

_menu_data: Optional[MenuData] = None


def set_menu_data(menu: MenuData):
    """Set the global menu data instance."""
    global _menu_data
    _menu_data = menu


def get_menu_data() -> Optional[MenuData]:
    """Get the global menu data instance."""
    return _menu_data


def get_all_items() -> list:
    """Legacy: Get all menu items from all restaurants."""
    if not _menu_data:
        return []

    items = []
    for restaurant in _menu_data.restaurants:
        for category in restaurant.get("categories", []):
            for item in category.get("items", []):
                items.append(
                    {
                        **item,
                        "categoryName": category.get("name"),
                        "restaurantId": restaurant.get("id"),
                    }
                )
    return items


def get_item_by_id(item_id: str) -> Optional[dict]:
    """Legacy: Get a specific menu item by ID."""
    if _menu_data:
        return _menu_data.get_item_by_id(item_id)
    return None


def find_item_by_name(name: str) -> Optional[dict]:
    """Legacy: Find a menu item by name."""
    if _menu_data:
        return _menu_data.find_item_by_name(name)
    return None
