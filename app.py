import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import datetime

def load_api_key():
    try:
        return st.secrets["API_KEY"]
    except KeyError:
        st.error("⚠️ API key not found in `.streamlit/secrets.toml`! Please add `GOOGLE_API_KEY` under `[general]`.")
        st.stop()

# Initialize LLM Model
def initialize_chat_model():
    api_key = load_api_key()
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

# Generate travel suggestion
def generate_travel_suggestions(source, destination, interest, budget, start_date, end_date, meal_pref, special_req):
    system_instruction = SystemMessage(
        content=("""
                You are an expert travel itinerary planner. Your job is to generate a well-structured, engaging, and visually appealing travel itinerary for a user based on their preferences.

                ### Guidelines for Itinerary Generation 
                - No Transportation Details 🚫 : Do not mention flights, trains, buses, or any transport mode.
                - Daily Breakdown  : Each day should be written in paragraph form, describing the experience in a flowing, engaging manner.
                - Balanced Schedule  : Ensure a mix of sightseeing, relaxation, cultural activities, and meals.
                - Personalization  : Tailor the itinerary to match the user's interests, budget, and dietary preferences.
                - Local Experience  : Recommend hidden gems, cultural experiences, and authentic local dining spots.

                ### Itinerary Format
                Day 1: [Title]   
                🌞 Morning : [Describe activities in an engaging way, e.g., "Start your day with a visit to..."].  
                🌅 Afternoon : [Describe afternoon activities with immersive details].  
                🌙 Evening : [Describe the evening experience, ensuring a balance of relaxation and fun].  

                Day 2: [Title]   
                🌞 Morning : [Activity description]  
                🌅 Afternoon : [Activity description]  
                🌙 Evening : [Activity description]  

                Final Notes 📝   
                - Provide essential travel tips.  
                - Mention any local phrases that might help.  
                - Recommend packing tips and cultural etiquette.  
                - End with a warm, engaging farewell message.

                Ensure the text flows naturally, creating an immersive experience for the traveler! 🚀
""")
    )
    
    user_request = HumanMessage(
        content=(f"""  # <-- f-string to allow variable interpolation
            Generate a detailed travel itinerary for a trip with the following details:

            - Source City: {source}
            - Destination: {destination}
            - Travel Dates: {start_date} to {end_date}
            - Budget: {budget} budget
            - Meal Preferences: {meal_pref}
            - Interests: {interest}
            - Special Requirements: {special_req or "None"}

            Ensure each day has different activities, covering the user’s interests.
            No transportation details should be included. Only list activities.
        """)
    )

    chat_model = initialize_chat_model()
    try:
        response = chat_model.invoke([system_instruction, user_request])
        return response.content if response else "⚠️ No response from AI."
    except Exception as err:
        return f"❌ Error fetching travel suggestions: {str(err)}"

# Streamlit UI
st.title("🌍 Smart AI Travel Planner")
st.markdown("Plan your journey effortlessly! Get AI-powered travel options with estimated costs and helpful insights.")

# User Inputs
start_point = st.text_input("📍 Departure Location", placeholder="E.g., Delhi")
end_point = st.text_input("🎯 Destination", placeholder="E.g., Kolkata")

interest = st.selectbox(
    "💡 Travel Interest",
    options=["ALL", "Culture", "Food", "Nature", "Shopping", "History", "Art", "Adventure", "Relaxation"]
)

budget = st.number_input("💰 Budget (in ₹)", min_value=0, step=500, format="%d")

start_date = st.date_input("🗓️ Start Date")
end_date = st.date_input("🗓️ End Date")

meal_pref = st.selectbox(
    "🍽️ Meal Preference",
    options=["No Preference", "Vegetarian", "Vegan", "Non-Vegetarian"]
)

special_req = st.text_area("📝 Other Special Requirements", placeholder="Wheelchair access, pets allowed, language preference, etc.")

# Button to Get Suggestions
if st.button("🔎 Get Travel Suggestions"):
    today = datetime.date.today()
    
    if not (start_point.strip() and end_point.strip()):
        st.warning("⚠️ Please enter both departure and destination locations.")
    
    elif start_date < today:
        st.warning("⚠️ Start date cannot be in the past.")
    
    elif end_date < start_date:
        st.warning("⚠️ End date cannot be earlier than start date.")
    
    if start_point.strip() and end_point.strip():
        with st.spinner("⏳ Gathering best travel options..."):
            travel_details = generate_travel_suggestions(
                start_point, end_point, interest, budget, start_date, end_date, meal_pref, special_req
            )
            st.success("✅ Here are your travel recommendations:")
            st.markdown(travel_details)

