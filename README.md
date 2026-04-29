# Destinaid AI - Smart Travel Recommender

Destinaid AI is a Flask-based smart travel recommendation web app that suggests destinations based on a user's travel goals, climate preference, budget, travel duration, and optional past trips. It combines a knowledge graph, weighted heuristic scoring, A* style ranking, itinerary generation, and an optional RNN-based preference layer to create personalized travel suggestions.

The project is intentionally lightweight on the frontend: the UI is built with HTML, CSS, Bootstrap, Font Awesome, and D3.js. The backend is written in Python with Flask, NumPy, and NetworkX.

## Features

- Personalized destination recommendations based on selected preferences.
- Click-to-toggle travel goal options for a simpler user experience.
- Climate and budget filtering.
- Match percentage scoring for each recommended destination.
- Destination cards with images and browser-side fallback handling.
- Itinerary generation for the selected destination and trip duration.
- Knowledge graph visualization using D3.js.
- Dedicated Knowledge Graph section accessible from the navigation bar.
- Vercel-friendly dependency setup with optional RNN behavior.
- Clean API endpoints for recommendations, itineraries, and graph data.

## Tech Stack

**Frontend**

- HTML5
- CSS3
- Bootstrap 5
- Font Awesome
- D3.js
- Vanilla JavaScript

**Backend**

- Python
- Flask
- NumPy
- NetworkX

## How The Recommendation System Works

Destinaid AI uses several layers of logic to recommend destinations.

## 1. Destination Dataset

The destination data is stored inside `load_destination_data()` in `app.py`. Each destination includes:

- `country`
- `description`
- `climate`
- `image`

Example:

```python
"Singapore": {
    "country": "Singapore",
    "description": "A city-state known for cleanliness, food, and shopping.",
    "climate": "warm",
    "image": "https://images.unsplash.com/photo-1530333716605-efb04f3a14a3"
}
```

## 2. Knowledge Graph

The app builds a graph where:

- Destination nodes represent places such as Bali, Paris, Tokyo, and Dubai.
- Feature nodes represent travel interests such as beach, culture, nightlife, food, history, shopping, mountains, and adventure.
- Edges connect destinations to features.
- Edge weights represent how strongly a destination matches a feature.

For example, Bali is strongly connected to:

- beach
- surfing
- warm
- relaxation
- culture

This graph allows the recommender to understand relationships between destinations and user interests.

## 3. User Preference Vector

When the user submits the form, the selected travel goals, climate, and budget are converted into a preference vector. Each selected feature receives a weight, so some features can influence recommendations more strongly than others.

Example feature weights:

```python
weights = {
    "beach": 1.2,
    "budget-friendly": 1.5,
    "culture": 1.3,
    "food": 1.4,
    "history": 1.3
}
```

## 4. Heuristic Scoring

Each destination is converted into its own feature vector. The app compares the user preference vector with each destination vector using a weighted dot product and a sigmoid normalization function.

The result is a match score between the user and each destination.

## 5. A* Style Ranking

The app uses a priority queue through Python's `heapq` module to rank destinations by score. The best matches are returned first.

The ranking also considers:

- Climate preference
- Budget preference
- Feature match strength

The API returns the top recommended destinations.

## 6. Optional RNN Preference Model

If the user enters at least three past trips, the app can train a small RNN model to learn from the sequence of past destinations.

## Local Setup

## 1. Clone Or Open The Project

Open the project folder:

```bash
cd "AI - Smart Travel Recommender"
```

## 2. Create A Virtual Environment

Windows:

```bash
python -m venv venv
```

macOS/Linux:

```bash
python3 -m venv venv
```

## 3. Activate The Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run The App

```bash
python app.py
```

The app should start at:

```text
http://127.0.0.1:5000
```
## Current Destinations

The app currently includes 26 locations :

- Bali
- Goa
- Phuket
- Jaipur
- Athens
- Barcelona
- Paris
- Rome
- New York
- Tokyo
- Sydney
- Cape Town
- Rio de Janeiro
- Dubai
- Singapore
- Istanbul
- London
- Amsterdam
- Venice
- Bangkok
- Marrakech
- San Francisco
- Los Angeles
- Vancouver
- Lisbon
- Seoul

## Main AI Concepts Used

## Knowledge Graphs

The app uses a graph structure to represent destinations and their relationships with travel features. This makes it easier to reason about why a destination matches a user's preferences.

## Heuristic Search

Each destination is scored using a heuristic function that compares user-selected interests with destination features. The highest-scoring places are returned as recommendations.

## Priority Queue Ranking

Python's `heapq` is used to efficiently rank and return the strongest destination matches.

## Optional RNN

The app includes optional RNN logic for learning from past trips. This can improve personalization when enough past trip data is provided.

## Future Improvements

- Add user accounts and saved trips.
- Add real pricing data or budget estimates.
- Add season-based recommendations.
- Add location-based distance scoring.
- Improve the itinerary generator with maps and time-aware routing

## License

This project is for educational and demonstration purposes.
