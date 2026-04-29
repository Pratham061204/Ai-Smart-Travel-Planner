import numpy as np
import networkx as nx
from flask import Flask, request, jsonify, render_template
import heapq

# Initialize Flask app
app = Flask(__name__, template_folder='.', static_folder='.')

# Define the Destinaid backend class
class DestinaidBackend:
    def __init__(self):
        self.model = None
        self.feature_weights = self.initialize_feature_weights()
        self.destination_data = self.load_destination_data()
        self.knowledge_graph = self.build_knowledge_graph()
        self.destination_to_index = {}  # For RNN model
        self.index_to_destination = {}  # For RNN model
        self.initialize_destination_mappings()
        
    def initialize_destination_mappings(self):
        """Create mappings between destinations and indices for the RNN model"""
        destinations = [n for n, attr in self.knowledge_graph.nodes(data=True) if attr.get('type') == 'destination']
        for i, dest in enumerate(destinations):
            self.destination_to_index[dest] = i
            self.index_to_destination[i] = dest
        
    def initialize_feature_weights(self):
        """Initialize default weights for features based on importance"""
        weights = {
            'beach': 1.2,
            'budget-friendly': 1.5,
            'surfing': 1.0,
            'culture': 1.3,
            'nightlife': 1.1,
            'warm': 1.0,
            'cold': 1.0,
            'temperate': 1.0,
            'adventure': 1.2,
            'relaxation': 1.1,
            'history': 1.3,
            'food': 1.4,
            'shopping': 1.1,
            'mountains': 1.2,
            'desert': 1.0,
            'urban': 1.1,
            'art': 1.0,
            'music': 1.0
        }
        return weights
    
    def load_destination_data(self):
        """Load additional destination data"""
        destinations_data = {
            "Bali": {"country": "Indonesia", "description": "A beautiful island known for beaches, surfing, and spiritual retreats.", "climate": "warm", "image": "https://images.unsplash.com/photo-1505228395891-9a51e7e86bf6"},
            "Goa": {"country": "India", "description": "Beach destination with Portuguese influence and vibrant nightlife.", "climate": "warm", "image": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2"},
            "Phuket": {"country": "Thailand", "description": "Thailand's largest island with beautiful beaches and nightlife.", "climate": "warm", "image": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?q=80&w=1101&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Jaipur": {"country": "India", "description": "The Pink City with rich history, palaces, and cultural experiences.", "climate": "warm", "image": "https://images.unsplash.com/photo-1602339752474-f77aa7bcaecd?q=80&w=1173&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Athens": {"country": "Greece", "description": "Ancient city with historical sites and Mediterranean cuisine.", "climate": "temperate", "image": "https://plus.unsplash.com/premium_photo-1661963145672-a2bd28eba0fb?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Barcelona": {"country": "Spain", "description": "Vibrant city with unique architecture, beaches, and food.", "climate": "warm", "image": "https://images.unsplash.com/photo-1523531294919-4bcd7c65e216"},
            "Paris": {"country": "France", "description": "City of lights, art, and romance.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34"},
            "Rome": {"country": "Italy", "description": "Historic city with ancient ruins and delicious cuisine.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1552832230-c0197dd311b5"},
            "New York": {"country": "USA", "description": "The city that never sleeps, full of culture and entertainment.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1499092346589-b9b6be3e94b2"},
            "Tokyo": {"country": "Japan", "description": "A bustling metropolis blending tradition and technology.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1542051841857-5f90071e7989"},
            "Sydney": {"country": "Australia", "description": "Famous for its harbor, beaches, and vibrant city life.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1624138784614-87fd1b6528f8?q=80&w=1333&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Cape Town": {"country": "South Africa", "description": "Known for its stunning landscapes and adventure activities.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1516026672322-bc52d61a55d5"},
            "Rio de Janeiro": {"country": "Brazil", "description": "Famous for its carnival, beaches, and lively culture.", "climate": "warm", "image": "https://images.unsplash.com/photo-1483729558449-99ef09a8c325"},
            "Dubai": {"country": "UAE", "description": "Modern city with luxury shopping and desert adventures.", "climate": "warm", "image": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c"},
            "Singapore": {"country": "Singapore", "description": "A city-state known for cleanliness, food, and shopping.", "climate": "warm", "image": "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?q=80&w=1052&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Istanbul": {"country": "Turkey", "description": "A city bridging Europe and Asia with rich history.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200"},
            "London": {"country": "UK", "description": "Historic city with museums, theaters, and royal palaces.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad"},
            "Amsterdam": {"country": "Netherlands", "description": "Known for canals, art museums, and vibrant nightlife.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4"},
            "Venice": {"country": "Italy", "description": "Romantic city famous for canals and architecture.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1533856493584-0c6ca8ca9ce3"},
            "Bangkok": {"country": "Thailand", "description": "Bustling city with street food, temples, and nightlife.", "climate": "warm", "image": "https://images.unsplash.com/photo-1563492065599-3520f775eeed"},
            "Marrakech": {"country": "Morocco", "description": "City with markets, palaces, and desert excursions.", "climate": "warm", "image": "https://images.unsplash.com/photo-1597212618440-806262de4f6b?q=80&w=1173&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "San Francisco": {"country": "USA", "description": "Known for the Golden Gate Bridge and tech culture.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29"},
            "Los Angeles": {"country": "USA", "description": "Entertainment capital with beaches and diverse culture.", "climate": "warm", "image": "https://images.unsplash.com/photo-1580655653885-65763b2597d0?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Vancouver": {"country": "Canada", "description": "City surrounded by mountains and ocean.", "climate": "cold", "image": "https://images.unsplash.com/photo-1517935706615-2717063c2225?q=80&w=1965&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"},
            "Lisbon": {"country": "Portugal", "description": "Coastal city with historic sites and great food.", "climate": "temperate", "image": "https://images.unsplash.com/photo-1585208798174-6cedd86e019a"},
            "Seoul": {"country": "South Korea", "description": "Modern city with rich culture and technology.", "climate": "temperate", "image": "https://plus.unsplash.com/premium_photo-1661886333708-877148b43ae1?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"}
        }
        return destinations_data

    def get_destination_image(self, destination):
        """Return a crop-friendly destination image URL with a real travel fallback."""
        image = self.destination_data.get(destination, {}).get('image')
        if image:
            separator = '&' if '?' in image else '?'
            return f"{image}{separator}auto=format&fit=crop&w=900&q=80"

        return self.get_fallback_image()

    def get_fallback_image(self):
        """Fallback image used when a destination has no image configured."""
        return "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=900&q=80"
        
    def build_knowledge_graph(self):
        """Build an enhanced knowledge graph of destinations and their features"""
        G = nx.Graph()
        
        # Add destinations and features
        destinations = [
            "Bali", "Goa", "Phuket", "Jaipur", "Athens", "Barcelona", "Paris", "Rome", 
            "New York", "Tokyo", "Sydney", "Cape Town", "Rio de Janeiro", "Dubai", 
            "Singapore", "Istanbul", "London", "Amsterdam", "Venice", "Bangkok", 
            "Marrakech", "San Francisco", "Los Angeles", "Vancouver", "Lisbon", "Seoul"
        ]
        
        features = [
            "beach", "budget-friendly", "surfing", "culture", "nightlife", "warm", 
            "cold", "temperate", "adventure", "relaxation", "history", "food", 
            "shopping", "mountains", "desert", "urban", "art", "music", "romantic",
            "architecture"
        ]
        
        # Add nodes with attributes
        for d in destinations:
            G.add_node(d, type='destination', **self.destination_data.get(d, {}))
            
        for f in features:
            G.add_node(f, type='feature')
        
        # Add edges (destination-feature relationships) with weights
        edges = [
            ("Bali", "beach", {'weight': 0.9}), ("Bali", "budget-friendly", {'weight': 0.7}), 
            ("Bali", "surfing", {'weight': 0.8}), ("Bali", "warm", {'weight': 0.9}), 
            ("Bali", "relaxation", {'weight': 0.8}), ("Bali", "culture", {'weight': 0.6}),
            ("Goa", "beach", {'weight': 0.8}), ("Goa", "budget-friendly", {'weight': 0.8}), 
            ("Goa", "nightlife", {'weight': 0.7}), ("Goa", "warm", {'weight': 0.9}),
            ("Phuket", "beach", {'weight': 0.9}), ("Phuket", "budget-friendly", {'weight': 0.7}), 
            ("Phuket", "nightlife", {'weight': 0.8}), ("Phuket", "warm", {'weight': 0.9}),
            ("Jaipur", "culture", {'weight': 0.9}), ("Jaipur", "history", {'weight': 0.9}), 
            ("Jaipur", "warm", {'weight': 0.8}), ("Jaipur", "architecture", {'weight': 0.8}),
            ("Athens", "culture", {'weight': 0.9}), ("Athens", "history", {'weight': 0.9}), 
            ("Athens", "food", {'weight': 0.8}), ("Athens", "temperate", {'weight': 0.8}),
            ("Barcelona", "beach", {'weight': 0.7}), ("Barcelona", "culture", {'weight': 0.9}), 
            ("Barcelona", "nightlife", {'weight': 0.8}), ("Barcelona", "food", {'weight': 0.8}), 
            ("Barcelona", "warm", {'weight': 0.7}), ("Barcelona", "architecture", {'weight': 0.9}),
            ("Paris", "culture", {'weight': 0.9}), ("Paris", "art", {'weight': 0.9}), 
            ("Paris", "shopping", {'weight': 0.8}), ("Paris", "temperate", {'weight': 0.8}),
            ("Rome", "culture", {'weight': 0.9}), ("Rome", "history", {'weight': 0.9}), 
            ("Rome", "food", {'weight': 0.8}), ("Rome", "temperate", {'weight': 0.8}),
            ("New York", "urban", {'weight': 0.9}), ("New York", "shopping", {'weight': 0.8}), 
            ("New York", "music", {'weight': 0.7}), ("New York", "temperate", {'weight': 0.8}),
            ("Tokyo", "urban", {'weight': 0.9}), ("Tokyo", "shopping", {'weight': 0.8}), 
            ("Tokyo", "culture", {'weight': 0.8}), ("Tokyo", "music", {'weight': 0.7}),
            ("Sydney", "beach", {'weight': 0.8}), ("Sydney", "urban", {'weight': 0.8}), 
            ("Sydney", "relaxation", {'weight': 0.7}), ("Sydney", "temperate", {'weight': 0.8}),
            ("Cape Town", "beach", {'weight': 0.7}), ("Cape Town", "mountains", {'weight': 0.8}), 
            ("Cape Town", "adventure", {'weight': 0.8}), ("Cape Town", "temperate", {'weight': 0.8}),
            ("Rio de Janeiro", "beach", {'weight': 0.8}), ("Rio de Janeiro", "nightlife", {'weight': 0.8}), 
            ("Rio de Janeiro", "warm", {'weight': 0.9}), ("Rio de Janeiro", "music", {'weight': 0.7}),
            ("Dubai", "desert", {'weight': 0.8}), ("Dubai", "shopping", {'weight': 0.9}), 
            ("Dubai", "urban", {'weight': 0.9}), ("Dubai", "warm", {'weight': 0.9}),
            ("Singapore", "urban", {'weight': 0.9}), ("Singapore", "shopping", {'weight': 0.9}), 
            ("Singapore", "food", {'weight': 0.8}), ("Singapore", "warm", {'weight': 0.9}),
            ("Istanbul", "culture", {'weight': 0.9}), ("Istanbul", "history", {'weight': 0.9}), 
            ("Istanbul", "food", {'weight': 0.8}), ("Istanbul", "temperate", {'weight': 0.8}),
            ("London", "culture", {'weight': 0.9}), ("London", "history", {'weight': 0.8}), 
            ("London", "music", {'weight': 0.7}), ("London", "temperate", {'weight': 0.8}),
            ("Amsterdam", "culture", {'weight': 0.8}), ("Amsterdam", "history", {'weight': 0.8}), 
            ("Amsterdam", "music", {'weight': 0.7}), ("Amsterdam", "temperate", {'weight': 0.8}),
            ("Venice", "culture", {'weight': 0.8}), ("Venice", "history", {'weight': 0.8}), 
            ("Venice", "romantic", {'weight': 0.9}), ("Venice", "temperate", {'weight': 0.8}),
            ("Bangkok", "urban", {'weight': 0.8}), ("Bangkok", "nightlife", {'weight': 0.8}), 
            ("Bangkok", "food", {'weight': 0.9}), ("Bangkok", "warm", {'weight': 0.9}),
            ("Marrakech", "desert", {'weight': 0.8}), ("Marrakech", "culture", {'weight': 0.8}), 
            ("Marrakech", "shopping", {'weight': 0.7}), ("Marrakech", "warm", {'weight': 0.9}),
            ("San Francisco", "urban", {'weight': 0.8}), ("San Francisco", "culture", {'weight': 0.8}), 
            ("San Francisco", "music", {'weight': 0.7}), ("San Francisco", "temperate", {'weight': 0.8}),
            ("Los Angeles", "urban", {'weight': 0.8}), ("Los Angeles", "beach", {'weight': 0.7}), 
            ("Los Angeles", "music", {'weight': 0.7}), ("Los Angeles", "warm", {'weight': 0.8}),
            ("Vancouver", "mountains", {'weight': 0.8}), ("Vancouver", "adventure", {'weight': 0.7}), 
            ("Vancouver", "cold", {'weight': 0.7}), ("Vancouver", "urban", {'weight': 0.8}),
            ("Lisbon", "beach", {'weight': 0.7}), ("Lisbon", "culture", {'weight': 0.8}), 
            ("Lisbon", "food", {'weight': 0.8}), ("Lisbon", "temperate", {'weight': 0.8}),
            ("Seoul", "urban", {'weight': 0.9}), ("Seoul", "culture", {'weight': 0.8}), 
            ("Seoul", "shopping", {'weight': 0.8}), ("Seoul", "temperate", {'weight': 0.8})
        ]
        
        G.add_edges_from([(u, v, d) for u, v, d in edges])
        
        # Add similarity edges between similar destinations
        similar_destinations = [
            ("Bali", "Phuket", {'weight': 0.8, 'type': 'similar'}),
            ("Bali", "Goa", {'weight': 0.7, 'type': 'similar'}),
            ("Paris", "Rome", {'weight': 0.7, 'type': 'similar'}),
            ("New York", "Tokyo", {'weight': 0.7, 'type': 'similar'}),
            ("Barcelona", "Lisbon", {'weight': 0.6, 'type': 'similar'}),
            ("London", "Amsterdam", {'weight': 0.6, 'type': 'similar'})
        ]
        G.add_edges_from(similar_destinations)
        
        return G

    def preprocess_user_preferences(self, prefs):
        """Convert user preferences into a feature vector for heuristic scoring"""
        features = [n for n, attr in self.knowledge_graph.nodes(data=True) if attr.get('type') == 'feature']
        vector = np.zeros(len(features))
        
        for i, f in enumerate(features):
            if f in prefs:
                # Use feature weights if available, otherwise default to 1
                vector[i] = self.feature_weights.get(f, 1.0)
                
        return features, vector

    def heuristic_function(self, user_vector, dest_vector, weights):
        """Enhanced heuristic suitability score with weighted feature match"""
        # Calculate base match score
        match_score = np.dot(user_vector, dest_vector * weights)
        
        # Apply sigmoid function to normalize score between 0 and 1
        normalized_score = 1 / (1 + np.exp(-match_score))
        
        return normalized_score

    def get_destination_feature_vector(self, destination, features):
        """Create weighted feature vector for destination"""
        if destination not in self.knowledge_graph:
            return np.zeros(len(features))
            
        dest_features = self.knowledge_graph[destination]
        vector = np.zeros(len(features))
        
        for i, f in enumerate(features):
            if f in dest_features:
                # Use edge weight if available, otherwise default to 1
                vector[i] = dest_features[f].get('weight', 1.0)
                
        return vector
        
    def a_star_search(self, user_vector, weights, climate_pref=None, budget_pref=None):
        """Enhanced A* search to rank destinations by suitability score"""
        features = [n for n, attr in self.knowledge_graph.nodes(data=True) if attr.get('type') == 'feature']
        heap = []
        
        for dest in [n for n, attr in self.knowledge_graph.nodes(data=True) if attr.get('type') == 'destination']:
            # Skip if climate preference doesn't match
            if climate_pref:
                dest_climate = self.knowledge_graph.nodes[dest].get('climate')
                if dest_climate and dest_climate != climate_pref:
                    continue
                    
            # Skip if budget preference doesn't match
            if budget_pref == "low" and "budget-friendly" not in self.knowledge_graph[dest]:
                continue
                
            dest_vector = self.get_destination_feature_vector(dest, features)
            match_score = self.heuristic_function(user_vector, dest_vector, weights)
            
            # Calculate additional penalties (could be enhanced with real data)
            cost_penalty = 0.1 if budget_pref == "low" and "budget-friendly" not in self.knowledge_graph[dest] else 0
            distance_penalty = 0  # Placeholder for actual distance calculation
            
            # Final score with penalties
            score = match_score - (cost_penalty + distance_penalty)
            
            heapq.heappush(heap, (-score, dest))  # max heap by negative score
            
        top_destinations = []
        for _ in range(min(5, len(heap))):  # Get top 5 destinations
            if heap:
                top_destinations.append(heapq.heappop(heap))
                
        return top_destinations

    def train_preference_model(self, past_trips, features):
        """Train an RNN model to predict preferences based on past trips"""
        if len(past_trips) < 3:  # Need at least 3 trips to train
            return None

        # The recommendation engine works without the optional RNN layer.
        # On serverless platforms like Vercel, TensorFlow may be unavailable
        # or too heavy to bundle, so we degrade gracefully.
        try:
            from keras.layers import SimpleRNN, Dense
            from keras.models import Sequential
            from keras.optimizers import Adam
            from keras.utils import to_categorical
        except Exception:
            return None
             
        # Convert destinations to indices
        try:
            trip_indices = [self.destination_to_index[trip] for trip in past_trips if trip in self.destination_to_index]
        except KeyError:
            return None
            
        if len(trip_indices) < 3:
            return None
            
        # Create sequences and labels
        X = []
        y = []
        sequence_length = 2  # Use 2 previous trips to predict the next
        
        for i in range(len(trip_indices) - sequence_length):
            X.append(trip_indices[i:i+sequence_length])
            y.append(trip_indices[i+sequence_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # One-hot encode the output
        num_destinations = len(self.destination_to_index)
        y = to_categorical(y, num_classes=num_destinations)
        
        # Reshape X for RNN input (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build RNN model
        model = Sequential([
            SimpleRNN(64, input_shape=(sequence_length, 1), activation='relu'),
            Dense(num_destinations, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        model.fit(X, y, epochs=50, batch_size=1, verbose=0)
        
        return model
        
    def predict_preferences(self, model, past_trips, features):
        """Predict user preferences based on past trips using RNN"""
        if model is None or len(past_trips) < 2:
            # Return default weights if not enough data
            return np.array([self.feature_weights.get(f, 1.0) for f in features])
            
        # Get the last two trips
        recent_trips = past_trips[-2:]
        
        # Convert to indices
        try:
            trip_indices = [self.destination_to_index[trip] for trip in recent_trips if trip in self.destination_to_index]
        except KeyError:
            return np.array([self.feature_weights.get(f, 1.0) for f in features])
            
        if len(trip_indices) < 2:
            return np.array([self.feature_weights.get(f, 1.0) for f in features])
            
        # Prepare input for prediction
        X = np.array(trip_indices).reshape(1, 2, 1)
        
        # Predict probabilities for next destination
        pred_probs = model.predict(X, verbose=0)[0]
        
        # Get feature vectors for all destinations
        feature_vectors = []
        for i in range(len(pred_probs)):
            dest = self.index_to_destination[i]
            feature_vectors.append(self.get_destination_feature_vector(dest, features))
            
        feature_vectors = np.array(feature_vectors)
        
        # Calculate weighted average of feature vectors
        weighted_features = np.dot(pred_probs, feature_vectors)
        
        # Ensure non-negative weights with minimum of 0.1
        return np.maximum(weighted_features, 0.1)

    def generate_recommendation_message(self, destination, user_prefs):
        """Generate a personalized recommendation message using destination features"""
        if destination not in self.knowledge_graph:
            return f"We recommend {destination} for your next trip!"
            
        # Get destination features and attributes
        dest_features = set(self.knowledge_graph[destination].keys())
        dest_attrs = self.knowledge_graph.nodes[destination]
        
        # Find matching preferences
        matching_prefs = [p for p in user_prefs if p in dest_features]
        
        # Build message
        message_parts = []
        message_parts.append(f"We think you'll love {destination}, {dest_attrs.get('country', 'a wonderful destination')}.")
        
        if matching_prefs:
            message_parts.append(f"It matches your preferences for {', '.join(matching_prefs)}.")
        else:
            message_parts.append("While it doesn't exactly match your current preferences, it offers a unique travel experience.")
            
        # Add description if available
        if 'description' in dest_attrs:
            message_parts.append(dest_attrs['description'])
            
        # Add specific highlights based on top features
        top_features = sorted(
            [(f, self.knowledge_graph[destination][f].get('weight', 1.0)) for f in dest_features],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_features:
            feature_names = [f[0] for f in top_features]
            message_parts.append(f"Top features include: {', '.join(feature_names)}.")
            
        return " ".join(message_parts)

    def generate_itinerary(self, destination, duration, interests):
        """Generate a detailed travel itinerary based on destination and interests"""
        # Get base itinerary for destination
        itinerary = self.get_base_itinerary(destination)
        
        # Adjust itinerary length to duration and prioritize based on interests
        prioritized_itinerary = self.prioritize_activities(itinerary, interests)
        days = min(int(duration), len(prioritized_itinerary))
        daily_plan = prioritized_itinerary[:days]
        
        # Format the itinerary with day numbers and time suggestions
        formatted_itinerary = []
        for i, activity in enumerate(daily_plan):
            time_suggestion = self.get_time_suggestion(i, activity)
            formatted_itinerary.append({
                "day": i + 1,
                "time": time_suggestion,
                "activity": activity,
                "duration": self.estimate_activity_duration(activity)
            })
            
        return formatted_itinerary
    
    def get_base_itinerary(self, destination):
        """Get base itinerary for a destination"""
        base_itineraries = {
            "Bali": [
                "Visit Uluwatu Temple and watch the Kecak dance performance",
                "Relax on Kuta Beach and try surfing lessons",
                "Explore Ubud Monkey Forest and art markets",
                "Take a day trip to Nusa Penida island",
                "Enjoy a traditional Balinese spa treatment",
                "Visit Tanah Lot Temple",
                "Hike Mount Batur for sunrise",
                "Explore rice terraces in Tegallalang",
                "Visit Bali Safari and Marine Park",
                "Take a cooking class",
                "Relax at Seminyak Beach",
                "Visit Goa Gajah (Elephant Cave)",
                "Explore local markets in Denpasar",
                "Enjoy seafood dinner at Jimbaran Bay",
                "Attend a traditional Balinese ceremony"
            ],
            "Goa": [
                "Relax on Anjuna Beach and visit the flea market",
                "Explore Old Goa's Portuguese architecture and churches",
                "Take a spice plantation tour",
                "Party at Baga Beach nightclubs",
                "Visit Dudhsagar Waterfalls",
                "Explore Panaji city",
                "Visit Basilica of Bom Jesus",
                "Enjoy water sports at Calangute Beach",
                "Take a river cruise on Mandovi River",
                "Visit Chapora Fort",
                "Explore local art galleries",
                "Attend a yoga retreat",
                "Visit Salim Ali Bird Sanctuary",
                "Enjoy Goan cuisine cooking class",
                "Relax at Palolem Beach"
            ],
            "Phuket": [
                "Island hopping tour to Phi Phi Islands",
                "Visit the Big Buddha and Wat Chalong Temple",
                "Explore Phuket Old Town's colorful streets",
                "Enjoy Patong Beach and nightlife",
                "Take a Thai cooking class",
                "Visit Phang Nga Bay and James Bond Island",
                "Relax at Kata or Karon Beach",
                "Explore Similan Islands National Park",
                "Visit Phuket Elephant Sanctuary",
                "Enjoy a traditional Thai massage",
                "Take a day trip to Coral Island",
                "Visit Phuket Fantasea cultural theme park",
                "Explore Sirinat National Park",
                "Take a sunset dinner cruise",
                "Visit Bangla Road for nightlife"
            ],
            "Jaipur": [
                "Visit the iconic Amber Fort and enjoy an elephant ride",
                "Explore the City Palace and its museums",
                "Visit Hawa Mahal (Palace of Winds)",
                "Explore Jantar Mantar observatory",
                "Shop at Johari Bazaar for jewelry and textiles",
                "Take a day trip to Chand Baori stepwell",
                "Visit Jaigarh Fort and its massive cannon",
                "Explore Albert Hall Museum",
                "Take a hot air balloon ride over the Pink City",
                "Visit Nahargarh Fort for sunset views",
                "Explore Galtaji Temple (Monkey Temple)",
                "Take a cooking class to learn Rajasthani cuisine",
                "Visit Birla Mandir and its marble architecture",
                "Explore Jal Mahal (Water Palace) from the shore",
                "Attend a traditional Rajasthani folk dance performance"
            ],
            "Athens": [
                "Visit the Acropolis and Parthenon",
                "Explore the Acropolis Museum",
                "Visit the Ancient Agora and Temple of Hephaestus",
                "Explore the National Archaeological Museum",
                "Wander through Plaka neighborhood",
                "Visit the Temple of Olympian Zeus",
                "Take a day trip to Cape Sounion and Temple of Poseidon",
                "Explore Monastiraki Flea Market",
                "Visit Syntagma Square and watch the changing of the guard",
                "Take a food tour of Greek cuisine",
                "Visit the Byzantine Museum",
                "Explore Lycabettus Hill for panoramic views",
                "Visit the Benaki Museum",
                "Take a day trip to Delphi",
                "Enjoy Greek wine tasting in Psiri neighborhood"
            ],
            "Barcelona": [
                "Visit Sagrada Familia and marvel at Gaudí's masterpiece",
                "Explore Park Güell and its colorful mosaics",
                "Stroll down La Rambla and visit La Boqueria Market",
                "Visit Casa Batlló and Casa Milà (La Pedrera)",
                "Explore the Gothic Quarter and Barcelona Cathedral",
                "Relax on Barceloneta Beach",
                "Visit Montjuïc Castle and Magic Fountain",
                "Explore the Picasso Museum",
                "Take a day trip to Montserrat",
                "Watch a FC Barcelona match at Camp Nou",
                "Visit the Barcelona Museum of Contemporary Art",
                "Take a cooking class to learn paella making",
                "Explore Tibidabo Amusement Park and Temple",
                "Visit Palau de la Música Catalana",
                "Enjoy tapas and flamenco show"
            ],
            "Paris": [
                "Visit the Eiffel Tower",
                "Explore the Louvre Museum",
                "Visit Notre-Dame Cathedral",
                "Stroll along the Champs-Élysées",
                "Visit Montmartre and Sacré-Cœur",
                "Explore the Palace of Versailles",
                "Visit Musée d'Orsay",
                "Take a Seine River cruise",
                "Visit the Panthéon",
                "Explore the Latin Quarter",
                "Visit Centre Pompidou",
                "Explore the Catacombs",
                "Visit Sainte-Chapelle",
                "Enjoy a day at Disneyland Paris",
                "Visit the Luxembourg Gardens"
            ],
            "Rome": [
                "Visit the Colosseum and Roman Forum",
                "Explore Vatican City, St. Peter's Basilica and the Vatican Museums",
                "Throw a coin in the Trevi Fountain",
                "Visit the Pantheon",
                "Explore Piazza Navona and its fountains",
                "Visit the Spanish Steps and shop on Via Condotti",
                "Explore the Borghese Gallery and Gardens",
                "Visit the Catacombs of Rome",
                "Take a food tour through Trastevere",
                "Visit Castel Sant'Angelo",
                "Explore the Appian Way and aqueducts",
                "Visit the Capitoline Museums",
                "Take a day trip to Ostia Antica",
                "Explore the Baths of Caracalla",
                "Visit the Mouth of Truth and Aventine Hill"
            ],
            "New York": [
                "Visit Times Square",
                "Explore Central Park",
                "Visit the Statue of Liberty and Ellis Island",
                "Tour the Metropolitan Museum of Art",
                "Walk across the Brooklyn Bridge",
                "Visit the 9/11 Memorial and Museum",
                "Explore the High Line",
                "Visit the Empire State Building",
                "Tour the American Museum of Natural History",
                "Explore Greenwich Village",
                "Visit Grand Central Terminal",
                "Attend a Broadway show",
                "Visit the Museum of Modern Art (MoMA)",
                "Explore Chinatown and Little Italy",
                "Take a helicopter tour over Manhattan"
            ],
            "Tokyo": [
                "Visit Senso-ji Temple in Asakusa",
                "Explore the Meiji Shrine and Yoyogi Park",
                "Experience the Shibuya Crossing",
                "Visit the Tokyo Skytree for panoramic views",
                "Explore Tsukiji Outer Market for fresh seafood",
                "Visit the Imperial Palace Gardens",
                "Shop in Ginza and Harajuku",
                "Experience teamLab Borderless digital art museum",
                "Visit Ueno Park and its museums",
                "Take a day trip to Nikko",
                "Explore Akihabara for electronics and anime",
                "Visit Shinjuku Gyoen National Garden",
                "Experience a traditional Japanese tea ceremony",
                "Visit Tokyo Disneyland or DisneySea",
                "Explore the nightlife in Roppongi or Shinjuku"
            ],
            "Sydney": [
                "Visit the Sydney Opera House",
                "Explore Sydney Harbour Bridge and consider the BridgeClimb",
                "Relax at Bondi Beach and walk the Bondi to Coogee coastal path",
                "Visit Taronga Zoo",
                "Explore the Royal Botanic Garden",
                "Take a ferry to Manly Beach",
                "Visit the Art Gallery of New South Wales",
                "Explore Darling Harbour and the SEA LIFE Sydney Aquarium",
                "Take a day trip to the Blue Mountains",
                "Visit the Queen Victoria Building for shopping",
                "Explore The Rocks historic district",
                "Take a Sydney Harbour cruise",
                "Visit the Australian Museum",
                "Explore Paddington Markets and boutiques",
                "Experience the nightlife in Kings Cross or Newtown"
            ],
            "Cape Town": [
                "Visit Table Mountain via cable car",
                "Explore the V&A Waterfront",
                "Visit Robben Island where Nelson Mandela was imprisoned",
                "Tour the Cape of Good Hope and Cape Point",
                "Visit Boulders Beach to see African penguins",
                "Explore Kirstenbosch National Botanical Garden",
                "Take a wine tour in Stellenbosch or Franschhoek",
                "Visit the District Six Museum",
                "Explore the colorful Bo-Kaap neighborhood",
                "Take a day trip to Hermanus for whale watching (in season)",
                "Visit the Two Oceans Aquarium",
                "Hike Lion's Head for sunrise or sunset",
                "Explore the Castle of Good Hope",
                "Visit Groot Constantia wine estate",
                "Relax on Camps Bay Beach"
            ],
            "Rio de Janeiro": [
                "Visit Christ the Redeemer statue",
                "Relax on Copacabana and Ipanema beaches",
                "Take the cable car to Sugarloaf Mountain",
                "Explore the historic Santa Teresa neighborhood",
                "Visit the Metropolitan Cathedral",
                "Explore Tijuca National Park",
                "Visit the Museum of Tomorrow",
                "Take a samba lesson",
                "Explore the Selarón Steps",
                "Visit the Maracanã Stadium",
                "Explore the Botanical Garden",
                "Take a boat tour of Guanabara Bay",
                "Visit the Museum of Modern Art",
                "Explore Lapa neighborhood and nightlife",
                "Take a hang gliding experience over the city"
            ],
            "Dubai": [
                "Visit the Burj Khalifa, the world's tallest building",
                "Explore the Dubai Mall and watch the fountain show",
                "Visit the Palm Jumeirah and Atlantis resort",
                "Experience a desert safari with dune bashing",
                "Explore the historic Al Fahidi district",
                "Visit the Dubai Museum in Al Fahidi Fort",
                "Shop at the Gold and Spice Souks",
                "Take an abra ride across Dubai Creek",
                "Visit the Dubai Frame",
                "Explore Miracle Garden (in season)",
                "Visit IMG Worlds of Adventure theme park",
                "Experience skiing at Ski Dubai in Mall of the Emirates",
                "Visit the Dubai Opera",
                "Explore Global Village (in season)",
                "Take a helicopter tour over Dubai"
            ],
            "Singapore": [
                "Visit Gardens by the Bay and the Supertree Grove",
                "Explore Marina Bay Sands and its SkyPark",
                "Visit Sentosa Island and its attractions",
                "Explore Singapore Zoo and Night Safari",
                "Visit the National Museum of Singapore",
                "Explore Chinatown and its temples",
                "Visit Little India and Arab Street",
                "Shop on Orchard Road",
                "Explore the Singapore Botanic Gardens",
                "Visit the ArtScience Museum",
                "Take a Singapore River cruise",
                "Explore Pulau Ubin island",
                "Visit the National Gallery Singapore",
                "Experience the nightlife at Clarke Quay",
                "Take a food tour of hawker centers"
            ],
            "Istanbul": [
                "Visit Hagia Sophia",
                "Explore the Blue Mosque (Sultan Ahmed Mosque)",
                "Visit Topkapi Palace and its harem",
                "Explore the Grand Bazaar",
                "Visit the Basilica Cistern",
                "Take a Bosphorus cruise",
                "Explore the Spice Bazaar",
                "Visit Dolmabahçe Palace",
                "Explore Taksim Square and Istiklal Street",
                "Visit the Istanbul Archaeological Museums",
                "Explore Süleymaniye Mosque",
                "Visit Chora Church (Kariye Museum)",
                "Take a Turkish bath (hamam) experience",
                "Visit Galata Tower",
                "Explore the Asian side of Istanbul"
            ],
            "London": [
                "Visit the British Museum",
                "Explore the Tower of London and see the Crown Jewels",
                "Visit Buckingham Palace and watch the Changing of the Guard",
                "Explore the Houses of Parliament and Big Ben",
                "Visit Westminster Abbey",
                "Ride the London Eye",
                "Explore the Victoria and Albert Museum",
                "Visit the Natural History Museum",
                "Explore Camden Market",
                "Visit St. Paul's Cathedral",
                "Explore Hyde Park and Kensington Gardens",
                "Visit the Tate Modern",
                "Explore Covent Garden",
                "Take a day trip to Windsor Castle",
                "Experience a West End show"
            ],
            "Amsterdam": [
                "Visit the Rijksmuseum",
                "Explore the Van Gogh Museum",
                "Take a canal cruise",
                "Visit the Anne Frank House",
                "Explore Vondelpark",
                "Visit the Heineken Experience",
                "Explore the Jordaan neighborhood",
                "Visit the Royal Palace of Amsterdam",
                "Explore the NEMO Science Museum",
                "Visit the Amsterdam Museum",
                "Explore the Albert Cuyp Market",
                "Visit the Rembrandt House Museum",
                "Take a day trip to Zaanse Schans windmills",
                "Explore the Red Light District",
                "Visit the Stedelijk Museum"
            ],
            "Venice": [
                "Explore St. Mark's Square and Basilica",
                "Visit Doge's Palace",
                "Take a gondola ride through the canals",
                "Explore the Rialto Bridge and Market",
                "Visit the Peggy Guggenheim Collection",
                "Explore the Grand Canal by vaporetto",
                "Visit the Gallerie dell'Accademia",
                "Explore Murano and Burano islands",
                "Visit the Basilica di Santa Maria della Salute",
                "Explore the Jewish Ghetto",
                "Visit Ca' Rezzonico",
                "Take a cicchetti tour (Venetian tapas)",
                "Visit the Scuola Grande di San Rocco",
                "Explore Lido Beach",
                "Visit the Naval History Museum"
            ],
            "Bangkok": [
                "Visit the Grand Palace and Wat Phra Kaew",
                "Explore Wat Pho and the Reclining Buddha",
                "Visit Wat Arun (Temple of Dawn)",
                "Explore Chatuchak Weekend Market",
                "Take a boat tour on the Chao Phraya River",
                "Visit Jim Thompson House",
                "Explore Chinatown (Yaowarat)",
                "Visit Lumphini Park",
                "Explore the Bangkok Art and Culture Centre",
                "Visit Wat Saket (Golden Mount)",
                "Explore Asiatique The Riverfront",
                "Visit Siam Paragon and other shopping malls",
                "Take a Thai cooking class",
                "Explore Khao San Road",
                "Visit the National Museum"
            ],
            "Marrakech": [
                "Explore Jemaa el-Fnaa square",
                "Visit the Bahia Palace",
                "Explore the Koutoubia Mosque",
                "Visit the Majorelle Garden",
                "Explore the Medina and its souks",
                "Visit the Saadian Tombs",
                "Explore the Ben Youssef Madrasa",
                "Visit the Marrakech Museum",
                "Take a day trip to the Atlas Mountains",
                "Visit the Menara Gardens",
                "Explore the Palmeraie by camel",
                "Visit El Badi Palace",
                "Take a hammam spa experience",
                "Explore Gueliz, the new city",
                "Take a cooking class to learn Moroccan cuisine"
            ],
            "San Francisco": [
                "Visit the Golden Gate Bridge",
                "Explore Alcatraz Island",
                "Visit Fisherman's Wharf and Pier 39",
                "Ride a cable car",
                "Explore Golden Gate Park",
                "Visit the Palace of Fine Arts",
                "Explore Chinatown",
                "Visit Lombard Street",
                "Explore the Mission District and its murals",
                "Visit the California Academy of Sciences",
                "Explore the Ferry Building Marketplace",
                "Visit Alamo Square and the Painted Ladies",
                "Explore the de Young Museum",
                "Visit Twin Peaks for panoramic views",
                "Take a day trip to Muir Woods"
            ],
            "Los Angeles": [
                "Visit the Hollywood Walk of Fame and TCL Chinese Theatre",
                "Explore Universal Studios Hollywood",
                "Visit the Griffith Observatory and Park",
                "Explore Santa Monica Pier and Beach",
                "Visit the Getty Center",
                "Explore Venice Beach and Boardwalk",
                "Visit the Los Angeles County Museum of Art (LACMA)",
                "Explore Beverly Hills and Rodeo Drive",
                "Visit the Warner Bros. Studio Tour",
                "Explore Downtown LA and the Arts District",
                "Visit the Natural History Museum",
                "Explore the Huntington Library and Gardens",
                "Visit the Petersen Automotive Museum",
                "Take a day trip to Disneyland",
                "Explore the Sunset Strip"
            ],
            "Vancouver": [
                "Explore Stanley Park and the Seawall",
                "Visit Granville Island Public Market",
                "Take the Capilano Suspension Bridge Park tour",
                "Visit the Vancouver Aquarium",
                "Explore Gastown and its Steam Clock",
                "Visit the Museum of Anthropology",
                "Explore Queen Elizabeth Park",
                "Visit Science World",
                "Take a day trip to Grouse Mountain",
                "Explore the Vancouver Art Gallery",
                "Visit VanDusen Botanical Garden",
                "Explore Kitsilano Beach",
                "Visit the Vancouver Lookout",
                "Take a day trip to Whistler",
                "Explore Chinatown and Dr. Sun Yat-Sen Classical Chinese Garden"
            ],
            "Lisbon": [
                "Visit São Jorge Castle",
                "Explore the Alfama district",
                "Visit Belém Tower",
                "Explore the Jerónimos Monastery",
                "Visit the National Tile Museum",
                "Take Tram 28 through historic neighborhoods",
                "Explore the LX Factory",
                "Visit the Gulbenkian Museum",
                "Explore the Bairro Alto district",
                "Visit the Lisbon Oceanarium",
                "Explore the Time Out Market",
                "Visit the National Coach Museum",
                "Take a day trip to Sintra",
                "Explore the Chiado district",
                "Visit the MAAT (Museum of Art, Architecture and Technology)"
            ],
            "Seoul": [
                "Visit Gyeongbokgung Palace",
                "Explore Bukchon Hanok Village",
                "Visit N Seoul Tower",
                "Explore Myeongdong shopping district",
                "Visit Changdeokgung Palace and Secret Garden",
                "Explore Insadong for traditional crafts",
                "Visit the National Museum of Korea",
                "Explore Hongdae area for youth culture",
                "Visit the War Memorial of Korea",
                "Explore Dongdaemun Design Plaza",
                "Visit Bongeunsa Temple",
                "Explore Gangnam district",
                "Visit the Leeum Samsung Museum of Art",
                "Take a day trip to the DMZ",
                "Explore Namdaemun Market"
            ]
        }
        
        # Return specific itinerary if available, otherwise generic
        if destination in base_itineraries:
            return base_itineraries[destination]
            
        # Generic itinerary for unknown destinations
        return [
            f"Day 1: Explore {destination}'s main attractions",
            f"Day 2: Visit museums and historical sites in {destination}",
            f"Day 3: Experience local cuisine and shopping",
            f"Day 4: Take a day trip to nearby attractions",
            f"Day 5: Relax and enjoy local entertainment",
            f"Day 6: Visit parks and natural attractions",
            f"Day 7: Explore less touristy neighborhoods",
            f"Day 8: Take a guided cultural tour",
            f"Day 9: Visit art galleries and cultural centers",
            f"Day 10: Experience nightlife and entertainment",
            f"Day 11: Take a cooking class or food tour",
            f"Day 12: Visit local markets and shops",
            f"Day 13: Relax at a spa or beach",
            f"Day 14: Take photos at scenic viewpoints",
            f"Day 15: Last-minute shopping and farewell dinner"
        ]
    
    def prioritize_activities(self, activities, interests):
        """Prioritize itinerary activities based on user interests"""
        if not interests:
            return activities
            
        # Simple keyword-based prioritization
        prioritized = []
        remaining = []
        
        for activity in activities:
            if any(interest.lower() in activity.lower() for interest in interests):
                prioritized.append(activity)
            else:
                remaining.append(activity)
                
        return prioritized + remaining
    
    def get_time_suggestion(self, day_index, activity):
        """Suggest a time of day for an activity"""
        if day_index == 0:  # First day
            return "Morning" if "relax" in activity.lower() else "Afternoon"
            
        if "beach" in activity.lower() or "relax" in activity.lower():
            return "Afternoon"
        elif "nightlife" in activity.lower() or "dinner" in activity.lower():
            return "Evening"
        elif "sunrise" in activity.lower():
            return "Early Morning"
        elif "museum" in activity.lower() or "historical" in activity.lower():
            return "Morning"
        else:
            return "Mid-day"
    
    def estimate_activity_duration(self, activity):
        """Estimate duration for an activity"""
        if "day trip" in activity.lower():
            return "Full day"
        elif "relax" in activity.lower() or "beach" in activity.lower():
            return "2-3 hours"
        elif "museum" in activity.lower() or "historical" in activity.lower():
            return "1-2 hours"
        elif "tour" in activity.lower():
            return "3-4 hours"
        else:
            return "1-2 hours"

    def get_knowledge_graph_data(self):
        """Convert knowledge graph to a format suitable for visualization"""
        nodes = []
        node_id_map = {}
        
        # Assign unique IDs to nodes for D3
        for i, (node, attr) in enumerate(self.knowledge_graph.nodes(data=True)):
            node_id_map[node] = i
            node_type = attr.get('type', 'unknown')
            group = 1 if node_type == 'destination' else 2
            nodes.append({
                "id": i,
                "name": node,
                "group": group,
                "type": node_type
            })
        
        links = []
        for source, target, data in self.knowledge_graph.edges(data=True):
            links.append({
                "source": node_id_map[source],
                "target": node_id_map[target],
                "value": data.get('weight', 1),
                "type": data.get('type', 'feature')
            })
        
        return {"nodes": nodes, "links": links}

# Initialize backend
backend = DestinaidBackend()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/knowledge-graph', methods=['GET'])
def knowledge_graph():
    return jsonify({
        'success': True,
        'knowledge_graph': backend.get_knowledge_graph_data()
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Extract user preferences with validation
        travel_goals = data.get('travel_goals', [])
        if not isinstance(travel_goals, list):
            return jsonify({'success': False, 'error': 'Invalid travel_goals format'}), 400
            
        climate = data.get('climate', '')
        budget = data.get('budget', '')
        duration = int(data.get('duration', 1))
        past_trips = data.get('past_trips', [])
        if not isinstance(past_trips, list):
            past_trips = [t.strip() for t in past_trips.split(',')] if past_trips else []

        # Build user preference features
        user_prefs = set(travel_goals)
        if climate:
            user_prefs.add(climate)
        if budget == "low":
            user_prefs.add("budget-friendly")

        # Preprocess user preferences
        features, user_vector = backend.preprocess_user_preferences(user_prefs)

        # Train preference model if past trips provided
        preference_model = backend.train_preference_model(past_trips, features)

        # Define weights for heuristic
        weights = np.ones(len(features))
        if preference_model is not None and past_trips:
            weights = backend.predict_preferences(preference_model, past_trips, features)

        # Use A* search to get top destinations with climate and budget filters
        top_destinations = backend.a_star_search(
            user_vector, 
            weights, 
            climate_pref=climate if climate else None,
            budget_pref=budget if budget else None
        )

        # Generate recommendation messages for each destination
        recommendations = []
        for score, dest in top_destinations:
            msg = backend.generate_recommendation_message(dest, list(user_prefs))
            dest_info = backend.destination_data.get(dest, {})
            
            recommendations.append({
                'destination': dest,
                'message': msg,
                'score': -score,  # Convert back to positive
                'country': dest_info.get('country', ''),
                'description': dest_info.get('description', ''),
                'image': backend.get_destination_image(dest)
            })

        # Get knowledge graph data for visualization
        knowledge_graph = backend.get_knowledge_graph_data()

        return jsonify({
            'success': True,
            'destinations': recommendations,
            'knowledge_graph': knowledge_graph
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/itinerary', methods=['POST'])
def itinerary():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        destination = data.get('destination')
        if not destination:
            return jsonify({'success': False, 'error': 'Destination required'}), 400
            
        duration = int(data.get('duration', 3))
        interests = data.get('interests', [])
        if not isinstance(interests, list):
            return jsonify({'success': False, 'error': 'Invalid interests format'}), 400

        # Generate itinerary
        itinerary_data = backend.generate_itinerary(destination, duration, interests)

        return jsonify({
            'success': True,
            'destination': destination,
            'duration': duration,
            'itinerary': itinerary_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
