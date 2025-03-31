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
            
            # Create table for user trips
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_trips (
                    user_name TEXT PRIMARY KEY,
                    trip_destinations TEXT,
                    trip_activities TEXT
                )
            ''')
            
            # Create table for recommended trips
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommended_trips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT,
                    destination TEXT,
                    explanation TEXT,
                    trip_details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_name) REFERENCES user_trips(user_name)
                )
            ''')
            
            # Create table for trip feedback
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trip_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT,
                    destination TEXT,
                    preferred TEXT,
                    hated TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_name) REFERENCES user_trips(user_name)
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

    def store_recommendation(self, user_name: str, destination: str, explanation: str, trip_details: dict):
        """Store a recommended trip for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert trip details to JSON string
            trip_details_json = json.dumps(trip_details)
            
            cursor.execute('''
                INSERT INTO recommended_trips (user_name, destination, explanation, trip_details)
                VALUES (?, ?, ?, ?)
            ''', (user_name, destination, explanation, trip_details_json))
            conn.commit()

    def get_user_recommendations(self, user_name: str, limit: int = 5) -> List[dict]:
        """Retrieve recent recommendations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT destination, explanation, trip_details, timestamp
                FROM recommended_trips
                WHERE user_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_name, limit))
            
            results = cursor.fetchall()
            recommendations = []
            for result in results:
                recommendations.append({
                    'destination': result[0],
                    'explanation': result[1],
                    'trip_details': json.loads(result[2]),
                    'timestamp': result[3]
                })
            return recommendations

    def store_trip_feedback(self, user_name: str, destination: str, preferred: List[str], hated: List[str]):
        """Store trip feedback in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert lists to JSON strings
            preferred_json = json.dumps(preferred)
            hated_json = json.dumps(hated)
            
            cursor.execute('''
                INSERT INTO trip_feedback (user_name, destination, preferred, hated)
                VALUES (?, ?, ?, ?)
            ''', (user_name, destination, preferred_json, hated_json))
            conn.commit()

    def get_user_feedback(self, user_name: str) -> List[dict]:
        """Retrieve feedback for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT destination, preferred, hated, timestamp
                FROM trip_feedback
                WHERE user_name = ?
                ORDER BY timestamp DESC
            ''', (user_name,))
            
            results = cursor.fetchall()
            feedback = []
            for result in results:
                feedback.append({
                    'destination': result[0],
                    'preferred': json.loads(result[1]),
                    'hated': json.loads(result[2]),
                    'timestamp': result[3]
                })
            return feedback 