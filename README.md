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

### `app.py`

Contains the Flask app and all backend recommendation logic.

Main responsibilities:

- Stores destination metadata such as country, description, climate, and image URL.
- Builds the knowledge graph using NetworkX.
- Scores destinations using user preferences.
- Filters results by climate and budget.
- Generates recommendation messages.
- Generates activity itineraries.
- Exposes API endpoints used by the frontend.

### `index.html`

Contains the entire frontend interface.

Main responsibilities:

- Displays the travel preference form.
- Sends recommendation requests to the backend.
- Displays recommended destination cards.
- Displays generated itineraries.
- Loads and renders the knowledge graph visualization.
- Handles image fallback behavior if a destination image fails to load.

### `requirements.txt`

Lists the Python dependencies needed for deployment and local setup.

```txt
Flask>=3.0,<4.0
numpy>=1.26,<3.0
networkx>=3.0,<4.0
```

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

You can manually update destination images by changing the `image` value for any destination.

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

If the user enters at least three past trips and Keras/TensorFlow is available, the app can train a small RNN model to learn from the sequence of past destinations.

This layer is optional. If Keras/TensorFlow is not installed, the app skips the RNN step and continues using the main knowledge graph and heuristic recommendation system.

This makes the app easier to deploy on platforms like Vercel, where large machine learning packages can be difficult to bundle.

## API Endpoints

## `GET /`

Renders the main web app.

## `POST /api/recommend`

Generates travel recommendations based on user preferences.

### Request Body

```json
{
  "travel_goals": ["beach", "food", "relaxation"],
  "climate": "warm",
  "budget": "low",
  "duration": 7,
  "past_trips": ["Bali", "Goa", "Phuket"]
}
```

### Response

```json
{
  "success": true,
  "destinations": [
    {
      "destination": "Bali",
      "message": "Bali is perfect for your beach, relaxation and warm preferences.",
      "score": 0.96,
      "country": "Indonesia",
      "description": "A beautiful island known for beaches, surfing, and spiritual retreats.",
      "image": "https://images.unsplash.com/photo-1505228395891-9a51e7e86bf6?auto=format&fit=crop&w=900&q=80"
    }
  ],
  "knowledge_graph": {
    "nodes": [],
    "links": []
  }
}
```

## `POST /api/itinerary`

Generates a travel itinerary for a selected destination.

### Request Body

```json
{
  "destination": "Bali",
  "duration": 7,
  "interests": ["beach", "relaxation", "food"]
}
```

### Response

```json
{
  "success": true,
  "destination": "Bali",
  "duration": 7,
  "itinerary": [
    {
      "day": 1,
      "activity": "Visit Uluwatu Temple",
      "time": "Morning",
      "duration": "1-2 hours"
    }
  ]
}
```

## `GET /api/knowledge-graph`

Returns graph data for the D3.js knowledge graph visualization.

### Response

```json
{
  "success": true,
  "knowledge_graph": {
    "nodes": [
      {
        "id": 0,
        "name": "Bali",
        "group": 1,
        "type": "destination"
      }
    ],
    "links": [
      {
        "source": 0,
        "target": 1,
        "value": 0.9,
        "type": "feature"
      }
    ]
  }
}
```

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

## Deployment Notes For Vercel

This project can be deployed as a Python Flask app on Vercel, but there are a few important notes:

- `requirements.txt` must be committed so Vercel can install dependencies.
- The app should avoid requiring TensorFlow at startup because TensorFlow can be too heavy for many serverless deployments.
- The RNN layer is optional, so the main recommender still works without TensorFlow.
- If recommendations fail on deployment, check the Vercel function logs for the `/api/recommend` route.
- The frontend now displays clearer API errors when the backend returns a non-JSON error page.

## Customizing Destinations

To add or edit destinations, open `app.py` and update `load_destination_data()`.

Each destination should follow this structure:

```python
"Destination Name": {
    "country": "Country Name",
    "description": "Short destination description.",
    "climate": "warm",
    "image": "https://example.com/image-url"
}
```

After adding a destination, also update the graph data inside `build_knowledge_graph()`:

- Add the destination name to the `destinations` list.
- Add feature edges for that destination in the `edges` list.
- Optionally add similarity edges to related destinations.

## Customizing Images

Destination images are stored in `app.py` inside `load_destination_data()`.

To replace an image, change the URL in the `image` field:

```python
"Singapore": {
    "country": "Singapore",
    "description": "A city-state known for cleanliness, food, and shopping.",
    "climate": "warm",
    "image": "YOUR_NEW_IMAGE_LINK_HERE"
}
```

The backend automatically appends Unsplash crop parameters to configured image URLs. The frontend also includes fallback behavior if an image fails to load.

## Current Destinations

The app currently includes:

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

The app includes optional RNN logic for learning from past trips. This can improve personalization when enough past trip data is provided and Keras/TensorFlow is installed.

## Future Improvements

- Add a database instead of storing destinations directly in code.
- Add user accounts and saved trips.
- Add real pricing data or budget estimates.
- Add real weather or season-based recommendations.
- Add location-based distance scoring.
- Improve the itinerary generator with maps and time-aware routing.
- Add admin tools for editing destinations and images.
- Add automated tests for the API endpoints.

## License

This project is for educational and demonstration purposes.
