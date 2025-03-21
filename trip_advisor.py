from datetime import datetime
from trip_memory import TripMemory

class TripAdvisor:
    def __init__(self, user_preferences, user_name):
        self.user_preferences = user_preferences
        self.user_name = user_name
        self.destinations = {
            'Bali': {
                'activities': ['explore', 'swim', 'relax', 'surf'],
                'climate': 'tropical',
                'budget_level': ['moderate', 'luxury'],
                'best_seasons': ['spring', 'summer'],
                'travel_styles': ['relaxed', 'cultural'],
                'min_budget': 1500,
                'activities_list': [
                    '🏖️ Premium beach clubs in Seminyak',
                    '🏄‍♂️ Surfing lessons in Canggu',
                    '🕉️ Temple visits in Ubud',
                    '🚶‍♂️ Rice terrace walks',
                    '🏊‍♂️ Snorkeling in Nusa Penida'
                ],
                'description': 'A perfect blend of culture and relaxation'
            },
            'Japan': {
                'activities': ['explore', 'eat', 'walk', 'shop'],
                'climate': 'moderate',
                'budget_level': ['moderate', 'luxury'],
                'best_seasons': ['spring', 'fall'],
                'travel_styles': ['busy', 'cultural'],
                'min_budget': 3000,
                'activities_list': [
                    '🏯 Historic temples and shrines',
                    '🍜 Food tours and cooking classes',
                    '🗻 Mount Fuji hiking',
                    '🚅 Bullet train experiences',
                    '🌸 Cherry blossom viewing (spring)'
                ],
                'description': 'Perfect for cultural immersion and modern experiences'
            },
            'New Zealand': {
                'activities': ['hike', 'explore', 'drive', 'adventure'],
                'climate': 'moderate',
                'budget_level': ['moderate', 'luxury'],
                'best_seasons': ['spring', 'fall'],
                'travel_styles': ['adventure', 'nature'],
                'min_budget': 2500,
                'activities_list': [
                    '🥾 Tongariro Alpine Crossing',
                    '🚴‍♂️ Mountain biking in Rotorua',
                    '🚗 Scenic drives along the South Island',
                    '🦁 Wildlife watching in Fiordland',
                    '🪂 Adventure sports in Queenstown'
                ],
                'description': 'Ideal for nature lovers and adventure seekers'
            }
        }

    def get_valid_input(self, prompt, input_type="text"):
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                print("I didn't catch that. Could you please provide an answer?")
                continue
            if input_type == "number":
                try:
                    return float(user_input)
                except ValueError:
                    print("Please enter a valid number.")
            else:
                return user_input

    def validate_date(self, date_str):
        try:
            date = datetime.strptime(date_str, '%m/%Y')
            if date < datetime.now():
                return False
            return True
        except ValueError:
            return False

    def get_travel_details(self):
        print(f"\nNow, let's plan your next adventure, {self.user_name}! 🌎")
        
        while True:
            when = self.get_valid_input("When are you thinking of traveling? (MM/YYYY): ")
            if self.validate_date(when):
                break
            print("Please enter a future date in the format MM/YYYY (e.g., 12/2024)")
        
        while True:
            budget = self.get_valid_input("What's your budget for this trip? (USD): ", "number")
            if budget > 0:
                break
            print("Please enter a valid budget amount greater than 0")
        
        while True:
            companions = self.get_valid_input("Who will you be traveling with? (solo/family/friends/partner): ").lower()
            if companions in ['solo', 'family', 'friends', 'partner']:
                break
            print("Please select one of: solo, family, friends, or partner")
        
        while True:
            duration = self.get_valid_input("How many days are you planning to stay? ", "number")
            if 1 <= duration <= 90:
                break
            print("Please enter a reasonable duration between 1 and 90 days")
        
        return {
            'when': when,
            'budget': budget,
            'companions': companions,
            'duration': duration
        }

    def calculate_destination_score(self, destination, attributes, travel_details):
        score = 0
        
        # Match activities
        for activity in attributes['activities']:
            score += self.user_preferences['activities'][activity]
        
        # Match travel style
        for style in attributes['travel_styles']:
            if style in self.user_preferences['travel_pace']:
                score += self.user_preferences['travel_pace'][style]
        
        # Budget compatibility
        if travel_details['budget'] >= attributes['min_budget']:
            score += 2
        
        # Season compatibility
        travel_date = datetime.strptime(travel_details['when'], '%m/%Y')
        season = self.get_season(travel_date.month)
        if season in attributes['best_seasons']:
            score += 1
        
        return score

    def get_season(self, month):
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'

    def recommend_destination(self, travel_details):
        print("\n🔍 Analyzing your travel preferences and requirements...")
        
        destination_scores = {}
        for dest, attributes in self.destinations.items():
            score = self.calculate_destination_score(dest, attributes, travel_details)
            destination_scores[dest] = score

        top_destinations = sorted(destination_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        print(f"\nBased on your travel style and preferences, {self.user_name}, I have two excellent recommendations for you!")
        
        for dest, score in top_destinations:
            self.present_recommendation(dest, travel_details)

    def present_recommendation(self, destination, travel_details):
        dest_info = self.destinations[destination]
        print(f"\n🌟 Recommended Destination: {destination}")
        print(f"\n{dest_info['description']}")
        
        print("\nWhy this matches your style:")
        matching_activities = set(dest_info['activities']) & set(self.user_preferences['activities'].keys())
        if matching_activities:
            print("- Aligns with your interest in:", ", ".join(matching_activities))
        
        if travel_details['budget'] >= dest_info['min_budget']:
            print(f"- Fits your budget of ${travel_details['budget']:,.2f}")
        
        print("\nRecommended activities:")
        for activity in dest_info['activities_list']:
            print(activity)
        
        print(f"\nBest time to visit: {', '.join(dest_info['best_seasons']).title()}")
        
        if 'cultural' in dest_info['travel_styles']:
            print("\n💡 Cultural Tip: Consider learning a few basic local phrases to enhance your experience!")
        
        if dest_info['climate'] == 'tropical':
            print("🌡️ Weather Tip: Pack light, breathable clothing and don't forget sun protection!")

def main():
    # First part: Collect travel history and build memory
    memory = TripMemory()
    user_preferences, user_name = memory.collect_past_trips()
    
    # Second part: Get travel recommendations
    advisor = TripAdvisor(user_preferences, user_name)
    travel_details = advisor.get_travel_details()
    advisor.recommend_destination(travel_details)

if __name__ == "__main__":
    main()
