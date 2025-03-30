import sqlite3
from typing import List, Tuple
import json

class TripDatabase:
    def __init__(self, db_path: str = "trip_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_trips (
                    user_name TEXT PRIMARY KEY,
                    trip_destinations TEXT,
                    trip_activities TEXT
                )
            ''')
            conn.commit()

    def update_user_trips(self, user_name: str, destinations: List[str], activities: List[str]):
        """Update or insert trip information for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert lists to JSON strings
            destinations_json = json.dumps(destinations)
            activities_json = json.dumps(activities)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_trips (user_name, trip_destinations, trip_activities)
                VALUES (?, ?, ?)
            ''', (user_name, destinations_json, activities_json))
            conn.commit()

    def get_user_trips(self, user_name: str) -> Tuple[List[str], List[str]]:
        """Retrieve trip information for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trip_destinations, trip_activities
                FROM user_trips
                WHERE user_name = ?
            ''', (user_name,))
            
            result = cursor.fetchone()
            if result:
                destinations = json.loads(result[0])
                activities = json.loads(result[1])
                return destinations, activities
            return [], [] 