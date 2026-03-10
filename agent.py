import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.callbacks.base import BaseCallbackHandler  # re-exported for app.py

load_dotenv(Path(__file__).parent / ".env")

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.4,
)

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_MAPS_KEY       = os.environ.get("GOOGLE_MAPS_API_KEY", "")
TRIPADVISOR_KEY       = os.environ.get("TRIPADVISOR_API_KEY", "")
AMADEUS_CLIENT_ID     = os.environ.get("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET", "")
AMADEUS_BASE          = "https://test.api.amadeus.com"
TAVILY_API_KEY        = os.environ.get("TAVILY_API_KEY", "")

# ── Amadeus token management ──────────────────────────────────────────────────
_amadeus_token: dict = {"access_token": None, "expires_at": 0.0}

def _get_amadeus_token() -> str:
    if _amadeus_token["access_token"] and time.time() < _amadeus_token["expires_at"] - 60:
        return _amadeus_token["access_token"]
    resp = requests.post(
        f"{AMADEUS_BASE}/v1/security/oauth2/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials",
              "client_id": AMADEUS_CLIENT_ID, "client_secret": AMADEUS_CLIENT_SECRET},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _amadeus_token["access_token"] = data["access_token"]
    _amadeus_token["expires_at"]   = time.time() + data.get("expires_in", 1799)
    return _amadeus_token["access_token"]

def _amadeus_headers() -> dict:
    return {"Authorization": f"Bearer {_get_amadeus_token()}"}

_IATA_FALLBACK = {
    "thessaloniki": "SKG", "sofia": "SOF", "athens": "ATH", "london": "LON",
    "paris": "PAR", "rome": "ROM", "milan": "MIL", "barcelona": "BCN",
    "madrid": "MAD", "amsterdam": "AMS", "berlin": "BER", "munich": "MUC",
    "frankfurt": "FRA", "vienna": "VIE", "zurich": "ZRH", "brussels": "BRU",
    "lisbon": "LIS", "oslo": "OSL", "stockholm": "STO", "copenhagen": "CPH",
    "helsinki": "HEL", "warsaw": "WAW", "prague": "PRG", "budapest": "BUD",
    "bucharest": "BUH", "istanbul": "IST", "dubai": "DXB", "new york": "NYC",
    "los angeles": "LAX", "chicago": "CHI", "miami": "MIA", "toronto": "YTO",
    "tokyo": "TYO", "singapore": "SIN", "bangkok": "BKK", "hong kong": "HKG",
    "sydney": "SYD", "cairo": "CAI", "johannesburg": "JNB",
}

def _resolve_city(city: str) -> dict:
    """Resolve city name → {iata, lat, lon}. Used for both flights and hotels."""
    key = city.lower().strip()
    if key in _IATA_FALLBACK:
        return {"iata": _IATA_FALLBACK[key], "lat": None, "lon": None}
    try:
        resp = requests.get(
            f"{AMADEUS_BASE}/v1/reference-data/locations",
            headers=_amadeus_headers(),
            params={"subType": "CITY", "keyword": city, "page[limit]": 1},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            item = data[0]
            geo  = item.get("geoCode", {})
            return {"iata": item.get("iataCode", ""), "lat": geo.get("latitude"), "lon": geo.get("longitude")}
    except Exception:
        pass
    return {"iata": city.upper()[:3], "lat": None, "lon": None}

def _city_to_iata(city: str) -> str:
    return _resolve_city(city)["iata"]

# ── TripAdvisor helpers ───────────────────────────────────────────────────────
TA_BASE = "https://api.content.tripadvisor.com/api/v1"

def _ta_headers():
    return {"accept": "application/json", "referer": "https://travelbot.app"}

def ta_available():
    return bool(TRIPADVISOR_KEY and TRIPADVISOR_KEY != "YOUR_TRIPADVISOR_KEY_HERE")

# ── Google Places helpers ─────────────────────────────────────────────────────
GPLACES_BASE = "https://maps.googleapis.com/maps/api/place"

def _gplace_photo_url(ref, max_width=400):
    return f"{GPLACES_BASE}/photo?maxwidth={max_width}&photoreference={ref}&key={GOOGLE_MAPS_KEY}"

def _static_map_url(address):
    enc = requests.utils.quote(address)
    return (f"https://maps.googleapis.com/maps/api/staticmap"
            f"?center={enc}&zoom=15&size=400x140&scale=2"
            f"&markers=color:red|{enc}&key={GOOGLE_MAPS_KEY}")

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_places(query: str, lat: str = "", lng: str = "") -> str:
    """Search for any places in a city: restaurants, cafes, bars, tourist attractions, sightseeing spots, museums, monuments, parks or landmarks.
    Query should include place type and location, e.g. 'restaurants in desired city', 'sightseeing in Athens', 'museums in Barcelona', 'tourist attractions in Rome'.
    If the user asks for nearby places (e.g. 'cafes near me', 'restaurants nearby', 'hotels close by'), pass the user's lat and lng coordinates from the location context.
    Returns a list of places with name, rating, address and map link."""
    result = _google_search(query, lat=lat, lng=lng)
    # Fall back to TripAdvisor if Google failed or returned nothing
    if result.startswith("Google Places search error") or result.startswith("No results"):
        if ta_available():
            return _ta_search(query, lat=lat, lng=lng)
    return result


def _ta_category(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["hotel", "accommodation", "hostel", "lodge", "stay", "resort", "airbnb"]):
        return "hotels"
    if any(w in q for w in ["attraction", "sightseeing", "museum", "monument", "park", "landmark",
                             "tour", "tourist", "sight", "gallery", "temple", "castle", "square"]):
        return "attractions"
    return "restaurants"


_TA_URL_TYPE = {"hotels": "Hotel_Review", "attractions": "Attraction_Review", "restaurants": "Restaurant_Review"}


def _ta_search(query: str, lat: str = "", lng: str = "") -> str:
    try:
        category = _ta_category(query)
        params = {
            "key":         TRIPADVISOR_KEY,
            "searchQuery": query,
            "category":    category,
            "language":    "en",
        }
        if lat and lng:
            params["latLong"]    = f"{lat},{lng}"
            params["radius"]     = 5
            params["radiusUnit"] = "km"
        resp = requests.get(
            f"{TA_BASE}/location/search",
            headers=_ta_headers(),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return f"No TripAdvisor results for '{query}'."
        url_type = _TA_URL_TYPE.get(category, "Restaurant_Review")
        # Extract city keyword from query to filter off-target results
        q_lower = query.lower()
        city_hint = ""
        for word in ["in ", "near ", "around "]:
            if word in q_lower:
                city_hint = q_lower.split(word, 1)[-1].strip().split()[0]
                break
        results = []
        for p in data[:10]:
            loc_id   = p.get("location_id", "")
            name     = p.get("name", "")
            addr     = p.get("address_obj", {})
            addr_str = ", ".join(filter(None, [addr.get("street1"), addr.get("city"), addr.get("country")]))
            # Skip results where address city doesn't match the target city (off-target name matches)
            addr_city = addr.get("city", "").lower()
            if city_hint and not lat and addr_city and city_hint not in addr_city:
                continue
            if len(results) >= 5:
                break
            results.append({
                "name":            name,
                "address":         addr_str,
                "rating":          p.get("rating", ""),
                "num_reviews":     p.get("num_reviews", ""),
                "location_id":     loc_id,
                "tripadvisor_url": f"https://www.tripadvisor.com/{url_type}-d{loc_id}",
                "map_url":         _static_map_url(addr_str) if addr_str else "",
                "maps_url":        f"https://www.google.com/maps/search/{requests.utils.quote(name + ' ' + addr_str)}",
                "source":          "tripadvisor",
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"TripAdvisor search error: {e}"


_google_places_available = True   # flips to False on first REQUEST_DENIED

def _google_search(query: str, lat: str = "", lng: str = "") -> str:
    global _google_places_available
    # Skip Google entirely if we already know billing is disabled
    if not _google_places_available:
        return "Google Places search error: API not available"
    try:
        if lat and lng:
            # Nearby Search enforces radius strictly (Text Search only biases)
            params = {
                "location": f"{lat},{lng}",
                "radius":   1000,          # hard 1 km limit
                "keyword":  query,
                "key":      GOOGLE_MAPS_KEY,
            }
            endpoint = f"{GPLACES_BASE}/nearbysearch/json"
        else:
            params   = {"query": query, "key": GOOGLE_MAPS_KEY}
            endpoint = f"{GPLACES_BASE}/textsearch/json"
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Detect billing disabled — mark Google as unavailable for this session
        if data.get("status") == "REQUEST_DENIED":
            _google_places_available = False
            return "Google Places search error: API billing disabled"
        results_raw = data.get("results", [])
        results = []
        price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
        for p in results_raw[:6]:
            photo_url = ""
            if p.get("photos"):
                photo_url = _gplace_photo_url(p["photos"][0]["photo_reference"])
            addr = p.get("formatted_address") or p.get("vicinity", "")
            results.append({
                "name":     p.get("name", ""),
                "address":  addr,
                "rating":   p.get("rating", "N/A"),
                "reviews":  p.get("user_ratings_total", 0),
                "price":    price_map.get(p.get("price_level"), ""),
                "open_now": p.get("opening_hours", {}).get("open_now"),
                "place_id": p.get("place_id", ""),
                "photo_url": photo_url,
                "map_url":  _static_map_url(addr),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id', '')}",
                "source":   "google",
            })
        return json.dumps(results, ensure_ascii=False) if results else f"No results found for '{query}'."
    except Exception as e:
        return f"Google Places search error: {e}"


@tool
def get_place_details(place_id: str) -> str:
    """Get detailed info about a specific place using its Google place_id.
    Returns phone number, website, opening hours, full address and OpenTable booking link if available."""
    try:
        resp = requests.get(
            f"{GPLACES_BASE}/details/json",
            params={
                "place_id": place_id,
                "key":      GOOGLE_MAPS_KEY,
                "fields":   "name,formatted_address,formatted_phone_number,website,opening_hours,rating,price_level,reviews,photos",
            },
            timeout=10,
        )
        resp.raise_for_status()
        r = resp.json().get("result", {})
        hours = r.get("opening_hours", {}).get("weekday_text", [])
        photo_url = ""
        if r.get("photos"):
            photo_url = _gplace_photo_url(r["photos"][0]["photo_reference"], 600)
        top_review = ""
        if r.get("reviews"):
            rv = r["reviews"][0]
            top_review = f"{rv.get('author_name', '')}: \"{rv.get('text', '')[:200]}\""
        name = r.get("name", "")
        return json.dumps({
            "name":          name,
            "address":       r.get("formatted_address", ""),
            "phone":         r.get("formatted_phone_number", ""),
            "website":       r.get("website", ""),
            "rating":        r.get("rating", ""),
            "opening_hours": hours,
            "photo_url":     photo_url,
            "top_review":    top_review,
            "maps_url":      f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            "opentable_url": f"https://www.opentable.com/s/?covers=2&term={requests.utils.quote(name)}",
        }, ensure_ascii=False)
    except Exception as e:
        return f"Place details error: {e}"


@tool
def get_place_reviews(place_name: str, city: str) -> str:
    """Get real visitor reviews for any restaurant, cafe, bar or attraction from TripAdvisor.
    Use this whenever the user asks for 'reviews', 'opinions', 'comments', 'what people say', 'мнения', 'отзиви'.
    This tool searches for the place automatically — no location_id needed.
    place_name: name of the place (e.g. 'Palermo', 'Acropolis Museum')
    city: city where the place is (e.g. 'Thessaloniki', 'Athens')"""
    if not ta_available():
        return "TripAdvisor API key not configured."
    try:
        # Step 1: find the location_id
        search = requests.get(
            f"{TA_BASE}/location/search",
            headers=_ta_headers(),
            params={"key": TRIPADVISOR_KEY, "searchQuery": f"{place_name} {city}", "language": "en"},
            timeout=10,
        )
        search.raise_for_status()
        data = search.json().get("data", [])
        if not data:
            return f"Could not find '{place_name}' in {city} on TripAdvisor."
        # Pick best match: prefer result where address city matches the requested city
        location_id = None
        found_name  = place_name
        for item in data:
            addr      = item.get("address_obj", {})
            item_city = addr.get("city", "").lower()
            if city.lower() in item_city or item_city in city.lower():
                location_id = item["location_id"]
                found_name  = item.get("name", place_name)
                break
        if not location_id:
            location_id = data[0]["location_id"]
            found_name  = data[0].get("name", place_name)

        # Step 2: fetch reviews
        rev = requests.get(
            f"{TA_BASE}/location/{location_id}/reviews",
            headers=_ta_headers(),
            params={"key": TRIPADVISOR_KEY, "language": "en", "limit": 5},
            timeout=10,
        )
        rev.raise_for_status()
        reviews = rev.json().get("data", [])
        if not reviews:
            return f"No English reviews found for {found_name}. Try asking in the local language."
        results = []
        for r in reviews:
            results.append({
                "rating":      r.get("rating"),
                "title":       r.get("title", ""),
                "text":        r.get("text", "")[:400],
                "author":      r.get("user", {}).get("username", "Anonymous"),
                "date":        r.get("published_date", "")[:10],
                "trip_type":   r.get("trip_type", ""),
            })
        return json.dumps({"place": found_name, "location_id": location_id, "reviews": results}, ensure_ascii=False)
    except Exception as e:
        return f"Reviews error: {e}"


@tool
def get_tripadvisor_reviews(location_id: str, language: str = "en") -> str:
    """Get the latest visitor reviews for a place from TripAdvisor using its location_id.
    Use get_place_reviews instead if you only have the place name.
    location_id: the TripAdvisor location_id from search results
    language: review language code, e.g. 'en', 'bg', 'de' (default 'en')"""
    if not ta_available():
        return "TripAdvisor API key not configured."
    try:
        resp = requests.get(
            f"{TA_BASE}/location/{location_id}/reviews",
            headers=_ta_headers(),
            params={"key": TRIPADVISOR_KEY, "language": language, "limit": 5},
            timeout=10,
        )
        resp.raise_for_status()
        reviews = resp.json().get("data", [])
        if not reviews:
            return f"No reviews found for location {location_id}."
        results = []
        for r in reviews:
            results.append({
                "rating":      r.get("rating"),
                "title":       r.get("title", ""),
                "text":        r.get("text", "")[:400],
                "author":      r.get("user", {}).get("username", "Anonymous"),
                "date":        r.get("published_date", "")[:10],
                "trip_type":   r.get("trip_type", ""),
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"TripAdvisor reviews error: {e}"


@tool
def get_tripadvisor_details(location_id: str) -> str:
    """Get TripAdvisor details, rating and reviews for a location by its TripAdvisor location_id."""
    if not ta_available():
        return "TripAdvisor API key not configured."
    try:
        resp = requests.get(
            f"{TA_BASE}/location/{location_id}/details",
            headers=_ta_headers(),
            params={"key": TRIPADVISOR_KEY, "language": "en", "currency": "USD"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        return json.dumps({
            "name":        d.get("name"),
            "rating":      d.get("rating"),
            "num_reviews": d.get("num_reviews"),
            "ranking":     d.get("ranking_data", {}).get("ranking_string"),
            "price":       d.get("price_level"),
            "cuisine":     [c["name"] for c in d.get("cuisine", [])],
            "hours":       d.get("hours", {}).get("weekday_text", []),
            "phone":       d.get("phone"),
            "website":     d.get("website"),
            "address":     d.get("address_obj", {}).get("address_string"),
            "ta_url":      d.get("web_url"),
        }, ensure_ascii=False)
    except Exception as e:
        return f"TripAdvisor details error: {e}"


@tool
def plan_trip(city: str, days: int = 3, interests: str = "") -> str:
    """Build a multi-day trip itinerary using TripAdvisor data for attractions AND restaurants.
    Use this tool whenever the user asks for a 'trip plan', 'itinerary', 'day guide', 'travel program',
    'what to do in X for N days', 'make me a plan', 'program for X days'.
    city: destination city (e.g. 'Thessaloniki', 'Rome', 'Sofia')
    days: number of days (default 3)
    interests: optional interests hint, e.g. 'history,food,beach'
    Returns top TripAdvisor attractions + restaurants combined, ready for itinerary planning."""
    if not ta_available():
        # Fallback to Google if TripAdvisor unavailable
        attractions = _google_search(f"top tourist attractions sightseeing in {city}")
        restaurants = _google_search(f"best restaurants in {city}")
        try:
            a_list = json.loads(attractions) if attractions.startswith("[") else []
            r_list = json.loads(restaurants) if restaurants.startswith("[") else []
        except Exception:
            a_list, r_list = [], []
        return json.dumps({"city": city, "days": days, "attractions": a_list, "restaurants": r_list}, ensure_ascii=False)

    query_suffix = f" {interests}" if interests else ""
    results = {"city": city, "days": days, "attractions": [], "restaurants": []}

    # ── Fetch attractions from TripAdvisor ──
    try:
        resp = requests.get(
            f"{TA_BASE}/location/search",
            headers=_ta_headers(),
            params={
                "key":         TRIPADVISOR_KEY,
                "searchQuery": f"top attractions sightseeing{query_suffix} in {city}",
                "category":    "attractions",
                "language":    "en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        for p in resp.json().get("data", [])[:6]:
            loc_id   = p.get("location_id", "")
            name     = p.get("name", "")
            addr     = p.get("address_obj", {})
            addr_str = ", ".join(filter(None, [addr.get("street1"), addr.get("city"), addr.get("country")]))
            results["attractions"].append({
                "name":            name,
                "address":         addr_str,
                "rating":          p.get("rating", ""),
                "num_reviews":     p.get("num_reviews", ""),
                "location_id":     loc_id,
                "tripadvisor_url": f"https://www.tripadvisor.com/Attraction_Review-d{loc_id}",
                "map_url":         _static_map_url(addr_str) if addr_str else "",
                "maps_url":        f"https://www.google.com/maps/search/{requests.utils.quote(name + ' ' + addr_str)}",
                "source":          "tripadvisor",
            })
    except Exception as e:
        results["attractions_error"] = str(e)

    # ── Fetch restaurants from TripAdvisor ──
    try:
        resp = requests.get(
            f"{TA_BASE}/location/search",
            headers=_ta_headers(),
            params={
                "key":         TRIPADVISOR_KEY,
                "searchQuery": f"best restaurants{query_suffix} in {city}",
                "category":    "restaurants",
                "language":    "en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        for p in resp.json().get("data", [])[:6]:
            loc_id   = p.get("location_id", "")
            name     = p.get("name", "")
            addr     = p.get("address_obj", {})
            addr_str = ", ".join(filter(None, [addr.get("street1"), addr.get("city"), addr.get("country")]))
            results["restaurants"].append({
                "name":            name,
                "address":         addr_str,
                "rating":          p.get("rating", ""),
                "num_reviews":     p.get("num_reviews", ""),
                "location_id":     loc_id,
                "tripadvisor_url": f"https://www.tripadvisor.com/Restaurant_Review-d{loc_id}",
                "map_url":         _static_map_url(addr_str) if addr_str else "",
                "maps_url":        f"https://www.google.com/maps/search/{requests.utils.quote(name + ' ' + addr_str)}",
                "source":          "tripadvisor",
            })
    except Exception as e:
        results["restaurants_error"] = str(e)

    return json.dumps(results, ensure_ascii=False)


def _is_overnight(segments: list) -> bool:
    """True if any segment or layover spans midnight."""
    for seg in segments:
        if seg["departure"]["at"][:10] != seg["arrival"]["at"][:10]:
            return True
    for i in range(len(segments) - 1):
        if segments[i]["arrival"]["at"][:10] != segments[i + 1]["departure"]["at"][:10]:
            return True
    return False

def _flight_extras(offer: dict) -> dict:
    """Extract baggage, cabin class and free meals from travelerPricings."""
    try:
        seg = offer["travelerPricings"][0]["fareDetailsBySegment"][0]
        bags   = seg.get("includedCheckedBags", {}).get("quantity", 0)
        baggage = f"{bags} x 23kg checked bag" if bags > 0 else "Hand luggage only"
        cabin  = seg.get("cabin", "ECONOMY").capitalize()
        free_meals = [a["description"] for a in seg.get("amenities", [])
                      if a.get("amenityType") == "MEAL" and not a.get("isChargeable")]
        food = ", ".join(free_meals) if free_meals else "Snacks/drinks for purchase"
        return {"baggage": baggage, "cabin": cabin, "food": food}
    except (KeyError, IndexError):
        return {"baggage": "N/A", "cabin": "Economy", "food": "N/A"}


@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   return_date: str = "", adults: int = 1, children: int = 0,
                   allow_overnight: bool = True) -> str:
    """Search for available flights between two cities using Amadeus.
    origin: departure city or airport (e.g. 'Sofia', 'LHR')
    destination: arrival city or airport (e.g. 'Paris', 'Rome')
    departure_date: YYYY-MM-DD
    return_date: YYYY-MM-DD for round trip (leave empty for one-way)
    adults: number of adult passengers (default 1)
    children: number of children 2-12 (default 0)
    allow_overnight: set False to filter out flights with overnight layovers (default True)
    Returns up to 5 flights with price, airline logo, baggage, cabin class and stops."""
    try:
        origin_code = _city_to_iata(origin)
        dest_code   = _city_to_iata(destination)
        params = {
            "originLocationCode":      origin_code,
            "destinationLocationCode": dest_code,
            "departureDate":           departure_date,
            "adults":                  adults,
            "max":                     10,
            "currencyCode":            "EUR",
        }
        if return_date:
            params["returnDate"] = return_date
        if children:
            params["children"] = children
        resp = requests.get(
            f"{AMADEUS_BASE}/v2/shopping/flight-offers",
            headers=_amadeus_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        all_offers = resp.json().get("data", [])
        if not all_offers:
            return f"No flights found from {origin} to {destination} on {departure_date}."

        # Apply overnight filter
        if not allow_overnight:
            all_offers = [o for o in all_offers
                          if not any(_is_overnight(itin["segments"]) for itin in o["itineraries"])]
        offers = all_offers[:5]

        results = []
        for offer in offers:
            price    = offer.get("price", {})
            total    = price.get("grandTotal") or price.get("total", "?")
            currency = price.get("currency", "EUR")
            itins    = offer.get("itineraries", [])
            carrier  = (offer.get("validatingAirlineCodes") or [""])[0]

            def _fmt(itin):
                segs = itin.get("segments", [])
                if not segs:
                    return {}
                dur = itin.get("duration", "").replace("PT","").replace("H","h ").replace("M","m").strip()
                return {
                    "from":       segs[0]["departure"]["iataCode"],
                    "to":         segs[-1]["arrival"]["iataCode"],
                    "departs":    segs[0]["departure"]["at"][:16].replace("T", " "),
                    "arrives":    segs[-1]["arrival"]["at"][:16].replace("T", " "),
                    "duration":   dur,
                    "stops":      len(segs) - 1,
                    "airlines":   ", ".join({s.get("carrierCode","") for s in segs}),
                    "is_overnight": _is_overnight(segs),
                }

            entry = {
                "price":      f"{total} {currency}",
                "carrier":    carrier,
                "logo":       f"https://s1.apideeplink.com/images/airlines/{carrier}.png",
                "seats_left": offer.get("numberOfBookableSeats", ""),
                "outbound":   _fmt(itins[0]) if itins else {},
                "extras":     _flight_extras(offer),
            }
            if len(itins) > 1:
                entry["return"] = _fmt(itins[1])
            results.append(entry)

        return json.dumps({
            "origin": origin_code, "destination": dest_code,
            "departure_date": departure_date, "return_date": return_date,
            "adults": adults, "children": children, "flights": results,
            "booking_url": f"https://www.google.com/travel/flights/search?q=flights+from+{origin}+to+{destination}+on+{departure_date}",
        }, ensure_ascii=False)
    except Exception as e:
        return f"Flight search error: {e}"


@tool
def search_hotels(city: str, check_in: str, check_out: str, adults: int = 1,
                  hotel_name: str = "", max_price: int = None, ratings: str = "",
                  board_type: str = "", amenities: str = "", best_rated: bool = False) -> str:
    """Search for available hotels in a city using Amadeus.
    city: destination city (e.g. 'Paris', 'London', 'Bansko', 'Barcelona')
    check_in: YYYY-MM-DD
    check_out: YYYY-MM-DD
    adults: number of guests (default 1)
    hotel_name: optional specific hotel name filter (e.g. 'Hilton', 'Marriott')
    max_price: maximum total price in EUR (integer, e.g. 200)
    ratings: star rating filter, comma-separated (e.g. '4,5' for 4 and 5-star hotels)
    board_type: meal plan — ROOM_ONLY, BREAKFAST, HALF_BOARD, FULL_BOARD, ALL_INCLUSIVE
    amenities: comma-separated amenity filter (e.g. 'SWIMMING_POOL,SPA,WIFI,FITNESS_CENTER,PARKING,PETS_ALLOWED')
    best_rated: if True, sorts results by highest guest rating first
    Returns up to 5 hotels with name, stars, room type, board basis, price and booking link."""
    try:
        city_info = _resolve_city(city)
        iata = city_info["iata"]
        lat  = city_info["lat"]
        lon  = city_info["lon"]

        # Step 1: get hotel list (by IATA or geocode for cities without airport)
        h_params: dict = {"radius": 15, "radiusUnit": "KM"}
        if ratings:
            h_params["ratings"] = ratings
        if amenities:
            h_params["amenities"] = amenities

        if iata:
            hotels_resp = requests.get(
                f"{AMADEUS_BASE}/v1/reference-data/locations/hotels/by-city",
                headers=_amadeus_headers(),
                params={"cityCode": iata, **h_params},
                timeout=10,
            )
        elif lat and lon:
            hotels_resp = requests.get(
                f"{AMADEUS_BASE}/v1/reference-data/locations/hotels/by-geocode",
                headers=_amadeus_headers(),
                params={"latitude": lat, "longitude": lon, **h_params},
                timeout=10,
            )
        else:
            return f"Could not resolve city '{city}'."

        hotels_resp.raise_for_status()
        hotel_list = hotels_resp.json().get("data", [])
        if not hotel_list:
            return f"No hotels found in {city}."

        # Filter by hotel name if specified
        if hotel_name:
            keyword = hotel_name.lower()
            matched = [h for h in hotel_list if keyword in h.get("name", "").lower()]
            hotel_list = matched if matched else hotel_list

        # Fewer hotel IDs = much faster Amadeus response (20 IDs = ~13s, 8 IDs = ~0.4s)
        id_limit = 3 if hotel_name else 8
        hotel_ids = ",".join(h["hotelId"] for h in hotel_list[:id_limit])

        # Step 2: get offers
        s_params: dict = {
            "hotelIds":    hotel_ids,
            "checkInDate": check_in,
            "checkOutDate": check_out,
            "adults":      adults,
            "currency":    "EUR",
            "bestRateOnly": True,
        }
        if max_price:
            s_params["priceRange"] = f"-{max_price}"
        if board_type:
            s_params["boardType"] = board_type
        if best_rated:
            s_params["sort"] = "RATING"

        offers_resp = requests.get(
            f"{AMADEUS_BASE}/v3/shopping/hotel-offers",
            headers=_amadeus_headers(),
            params=s_params,
            timeout=15,
        )
        offers_resp.raise_for_status()
        offers_data = offers_resp.json().get("data", [])
        if not offers_data:
            return f"No hotel availability in {city} for {check_in} to {check_out}."

        max_results = 1 if hotel_name else 5
        results = []
        for item in offers_data[:max_results]:
            hotel    = item.get("hotel", {})
            offers   = item.get("offers", [{}])
            offer    = offers[0] if offers else {}
            price    = offer.get("price", {})
            total    = price.get("total") or price.get("base", "?")
            currency = price.get("currency", "EUR")
            name     = hotel.get("name", "Unknown Hotel")
            enc      = requests.utils.quote(f"{name} {city}")
            room     = offer.get("room", {})
            room_cat = room.get("typeEstimated", {}).get("category", "Standard").replace("_", " ").capitalize()
            board    = offer.get("boardType", "").replace("_", " ").capitalize() or "Room only"
            results.append({
                "name":      name,
                "stars":     hotel.get("rating", ""),
                "address":   hotel.get("address", {}).get("lines", [""])[0],
                "price":     f"{total} {currency} total",
                "room_type": room_cat,
                "board":     board,
                "check_in":  check_in,
                "check_out": check_out,
                "nights":    (datetime.strptime(check_out, "%Y-%m-%d") - datetime.strptime(check_in, "%Y-%m-%d")).days,
                "map_url":   _static_map_url(f"{name} {city}"),
                "maps_url":  f"https://www.google.com/maps/search/{enc}",
                "book_url":  f"https://www.booking.com/search.html?ss={requests.utils.quote(name + ' ' + city)}",
            })
        return json.dumps({"city": city, "check_in": check_in, "check_out": check_out,
                           "adults": adults, "hotels": results}, ensure_ascii=False)
    except Exception as e:
        return f"Hotel search error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for real-time travel information not available in other tools.
    Use for: current travel advisories, visa requirements, local events, weather forecasts,
    entry restrictions, safety tips, best time to visit, currency/costs, cultural tips.
    Examples: 'visa requirements for Bulgaria 2026', 'is it safe to travel to X', 'events in Rome March 2026'.
    query: search query in English"""
    if not TAVILY_API_KEY:
        return "Web search not available (API key not configured)."
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":      TAVILY_API_KEY,
                "query":        query,
                "search_depth": "basic",
                "max_results":  5,
                "include_answer": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data    = resp.json()
        answer  = data.get("answer", "")
        results = data.get("results", [])
        snippets = [
            {"title": r.get("title"), "content": r.get("content", "")[:300]}
            for r in results[:5]
        ]
        return json.dumps({"answer": answer, "results": snippets}, ensure_ascii=False)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def book_restaurant(restaurant_name: str, location: str, date: str = "", party_size: str = "2") -> str:
    """Generate booking options for a restaurant via TheFork and Quandoo widgets.
    restaurant_name: name of the restaurant
    location: city or address
    date: preferred date (optional)
    party_size: number of people (default 2)"""
    name_enc = requests.utils.quote(restaurant_name)
    city_enc = requests.utils.quote(location)
    q_enc    = requests.utils.quote(f"{restaurant_name} {location}")
    date_str = f" on {date}" if date else ""
    return json.dumps({
        "message":     f"Booking options for {restaurant_name}{date_str} (party of {party_size}):",
        "thefork_url": f"https://www.thefork.com/search?query={name_enc}+{city_enc}",
        "quandoo_url": f"https://www.quandoo.com/en/result?q={name_enc}&city={city_enc}",
        "gmaps_url":   f"https://www.google.com/maps/search/{q_enc}",
        "restaurant":  restaurant_name,
        "location":    location,
        "date":        date,
        "party_size":  party_size,
        "action":      "show_booking_modal",
    }, ensure_ascii=False)


@tool
def get_directions(origin: str, destination: str) -> str:
    """Get Google Maps directions link from origin to destination.
    origin: starting point (address or city)
    destination: where to go (restaurant name + city)"""
    enc_o = requests.utils.quote(origin)
    enc_d = requests.utils.quote(destination)
    return json.dumps({
        "directions_url": f"https://www.google.com/maps/dir/{enc_o}/{enc_d}",
        "message": f"Directions from {origin} to {destination}",
    })


@tool
def get_transit_directions(origin: str, destination: str, mode: str = "transit") -> str:
    """Get real route options from Google Maps Directions API with duration and steps.
    Use this when user asks HOW to get from A to B, what transport to use, bus/metro/train routes,
    travel time between places, or 'how do I reach X from Y'.
    origin: starting point — address, place name or 'current location'
    destination: end point — address or place name
    mode: 'transit' (bus/metro/train), 'walking', 'driving', or 'bicycling' (default: transit)
    Returns route steps, duration, transit line names and a Google Maps link."""
    try:
        params = {
            "origin":      origin,
            "destination": destination,
            "mode":        mode,
            "key":         GOOGLE_MAPS_KEY,
            "language":    "en",
        }
        if mode == "transit":
            params["transit_mode"] = "bus|subway|train|tram"
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data   = resp.json()
        status = data.get("status")
        if status != "OK":
            return json.dumps({"error": f"No route found ({status}). Try being more specific."})

        route = data["routes"][0]["legs"][0]
        duration = route.get("duration", {}).get("text", "")
        distance = route.get("distance", {}).get("text", "")

        # Parse steps
        steps = []
        for s in route.get("steps", []):
            step = {
                "instruction": s.get("html_instructions", "").replace("<b>", "").replace("</b>", "")
                                 .replace("<div style=\"font-size:0.9em\">", " — ").replace("</div>", ""),
                "duration":    s.get("duration", {}).get("text", ""),
                "distance":    s.get("distance", {}).get("text", ""),
                "mode":        s.get("travel_mode", "").lower(),
            }
            if s.get("transit_details"):
                td = s["transit_details"]
                line = td.get("line", {})
                step["transit_line"]      = line.get("short_name") or line.get("name", "")
                step["transit_vehicle"]   = line.get("vehicle", {}).get("type", "").lower()
                step["departure_stop"]    = td.get("departure_stop", {}).get("name", "")
                step["arrival_stop"]      = td.get("arrival_stop", {}).get("name", "")
                step["num_stops"]         = td.get("num_stops", "")
            steps.append(step)

        enc_o = requests.utils.quote(origin)
        enc_d = requests.utils.quote(destination)
        return json.dumps({
            "origin":         origin,
            "destination":    destination,
            "mode":           mode,
            "duration":       duration,
            "distance":       distance,
            "steps":          steps,
            "directions_url": f"https://www.google.com/maps/dir/{enc_o}/{enc_d}",
        }, ensure_ascii=False)
    except Exception as e:
        return f"Transit directions error: {e}"


# ── Google Calendar ───────────────────────────────────────────────────────────
@tool
def add_to_calendar(title: str, date: str, time: str = "19:00",
                    duration_hours: int = 2, location: str = "", description: str = "") -> str:
    """Add a restaurant booking or travel activity to Google Calendar.
    Generates a link the user can click to save the event — no sign-in needed from the bot.
    title: event name, e.g. 'Dinner at Mykonos Restaurant'
    date: YYYY-MM-DD format
    time: start time HH:MM 24h (default 19:00)
    duration_hours: length of event in hours (default 2)
    location: restaurant address or city
    description: optional booking notes"""
    try:
        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt   = start_dt + timedelta(hours=int(duration_hours))
        fmt      = "%Y%m%dT%H%M%S"
        dates    = f"{start_dt.strftime(fmt)}/{end_dt.strftime(fmt)}"
        cal_url  = (
            "https://calendar.google.com/calendar/render?action=TEMPLATE"
            f"&text={requests.utils.quote(title)}"
            f"&dates={dates}"
            f"&location={requests.utils.quote(location)}"
            f"&details={requests.utils.quote(description)}"
            "&sf=true&output=xml"
        )
        return json.dumps({
            "action":       "calendar",
            "title":        title,
            "date":         date,
            "time":         time,
            "location":     location,
            "calendar_url": cal_url,
        })
    except Exception as e:
        return f"Calendar error: {e}"


# ── Callback: capture place results from tool outputs ─────────────────────────
class PlacesCapture(BaseCallbackHandler):
    def __init__(self):
        self.places = []

    def on_tool_end(self, output, **kwargs):
        try:
            data = json.loads(output)
            if isinstance(data, list) and data and "name" in data[0]:
                self.places.extend(data)
            elif isinstance(data, dict):
                # plan_trip returns {"attractions": [...], "restaurants": [...]}
                # search_hotels returns {"hotels": [...]}
                for key in ("attractions", "restaurants", "hotels"):
                    items = data.get(key, [])
                    if isinstance(items, list) and items and "name" in items[0]:
                        self.places.extend(items)
            self.places = self.places[-18:]  # keep max 18 for itineraries
        except Exception:
            pass


# ── Callback: capture calendar events from tool outputs ──────────────────────
class CalendarCapture(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_tool_end(self, output, **kwargs):
        try:
            data = json.loads(str(output))  # str() handles ToolMessage objects
            if isinstance(data, dict) and data.get("action") == "calendar":
                self.events.append(data)
        except Exception:
            pass


# ── Callback: capture tool names used ────────────────────────────────────────
_TOOL_LABELS = {
    "search_places":          "🔍 Places search",
    "plan_trip":              "🗺️ Trip planner",
    "get_place_details":      "📍 Place details",
    "get_place_reviews":      "⭐ Reviews",
    "get_tripadvisor_reviews":"⭐ Reviews",
    "search_flights":         "✈️ Flights search",
    "search_hotels":          "🏨 Hotels search",
    "web_search":             "🌐 Web search",
    "book_restaurant":        "📅 Restaurant booking",
    "get_directions":         "🗺️ Directions",
    "get_transit_directions": "🚌 Transit directions",
    "add_to_calendar":        "📅 Calendar",
}

class ToolsUsedCapture(BaseCallbackHandler):
    def __init__(self):
        self.tools = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "")
        label = _TOOL_LABELS.get(name, name)
        if label and label not in self.tools:
            self.tools.append(label)


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly and knowledgeable travel assistant. You can help with anything travel-related.

You help users:
- Plan multi-day city guides and itineraries (e.g. "3-day guide to Thessaloniki")
- Find restaurants, cafes, bars — including budget tips, local favourites, cuisine types
- Discover tourist attractions, sightseeing spots, museums, monuments, parks, day trips
- Answer general travel questions: transport, weather, visa tips, neighbourhoods, culture, budget advice
- Book restaurants or get contact details
- Show TripAdvisor reviews: use get_place_reviews(place_name, city) — it handles the search automatically. Always show whatever reviews come back, even if fewer than requested.
- Get directions between places
- Get real transit/bus/metro/walking routes with duration and steps: use get_transit_directions(origin, destination, mode). Present results clearly: total time, each step with mode icon (🚶 walking, 🚌 bus, 🚇 metro, 🚆 train), line name, stops. Always include the Google Maps link at the end.
- Add restaurant bookings or travel activities to Google Calendar (use add_to_calendar when user asks to "add to calendar", "save this", "book" with a date). After calling the tool, just confirm the event was created — do NOT include any URL or link in your text, the UI shows the calendar button automatically.
- Search for flights: use search_flights(origin, destination, departure_date, return_date, adults, children, allow_overnight). Present results clearly: price, airline logo, baggage allowance, cabin class, departure/arrival times, duration, stops. Mention one-way or round trip. Include the booking link.
- Search for hotels: use search_hotels(city, check_in, check_out, adults, hotel_name, max_price, ratings, board_type, amenities, best_rated). Works for cities without airports too (e.g. Bansko). Present each hotel with name, star rating, room type, board basis, total price and a booking link. Note total nights.
- Search the web for real-time info: use web_search(query) for visa requirements, travel advisories, current events, safety info, entry restrictions, local tips or anything time-sensitive not covered by other tools.

When asked for a day guide, itinerary, trip program or travel plan:
- ALWAYS use plan_trip(city, days) first — it fetches TripAdvisor attractions AND restaurants in one call.
- Structure the response clearly: Day 1, Day 2, etc.
- Mix morning sightseeing (from attractions list), lunch spots, afternoon activities and dinner recommendations (from restaurants list)
- Keep it practical and mobile-friendly
- Do NOT call search_places separately for attractions and restaurants when plan_trip was already called — use its results.

IMPORTANT: Always use the search_places tool to find real places — never say "no results" without trying it first.
For sightseeing, pass the query as-is, e.g. "tourist attractions in Thessaloniki" or "best restaurants in Athens city centre".

CONTEXT: You have full conversation history. If the user already mentioned a city or destination earlier in the chat, use it automatically for follow-up requests like "make me a 3-day plan" or "show me restaurants" — never ask which city if it was already mentioned.

When presenting individual places include:
- Name and rating (⭐)
- Price level if applicable ($ to $$$$)
- Whether it's open now (if known)

Format results with emojis and clear structure. Be concise and mobile-friendly.
If search returns JSON data, present it nicely — don't show raw JSON to the user.

CRITICAL: Do NOT include URLs, markdown links [text](url), or image syntax ![](url) in your text responses.
The UI already shows maps, photos and links automatically via place cards. Just mention place names and details in plain text."""

tools = [plan_trip, search_places, get_place_details, get_place_reviews, get_tripadvisor_reviews, search_flights, search_hotels, web_search, book_restaurant, get_directions, get_transit_directions, add_to_calendar]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ── Session store & agent factory ─────────────────────────────────────────────
_sessions: dict = {}

def get_agent(session_id: str) -> AgentExecutor:
    if session_id not in _sessions:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=10
        )
        agent = create_openai_tools_agent(llm, tools, prompt)
        _sessions[session_id] = AgentExecutor(
            agent=agent, tools=tools, memory=memory,
            verbose=False, max_iterations=20, handle_parsing_errors=True,
        )
    return _sessions[session_id]
