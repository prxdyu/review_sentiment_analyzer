# importing the required libraries
import streamlit as st
import pickle
import tensorflow as tf 
from model import predict

st.title("Hotel Review Sentiment Analyzer")

local_image_path = "images/images.png"
st.image(local_image_path, caption="",width=500)
    
st.write("""
        ### Analyze Sentiment of Reviews with Aspect-Based Sentiment Analysis
        ##### Gain insights into user sentiments about specific aspects of hotels or restaurants.
        """)
st.write("""
        #### How it works:
        1. Enter a review about a hotel or restaurant.
        2. Select the aspect you want to analyze (e.g., food quality, service, ambiance).
        3. Our model will predict the sentiment for the chosen aspect.
        4. Gain valuable insights into user opinions and experiences.
        """)
st.write("""
        #### Key Features:
        - Accurate sentiment analysis for specific aspects.
        - Easy-to-use interface for input and analysis.
        - Explore user sentiments in-depth.
        """)

# Textbox input for review
review = st.text_area("##### Enter your Review about the restaurant here:")

# edge cases for reviews
if not review.strip():
    st.error("Please enter a review.")
elif len(review.split(" "))<2:
    st.error("Please enter a review with atleast 2 words ")


# Text input for aspect
aspect = st.text_input("##### Enter the aspect (e.g., food quality, service, ambiance):",  max_chars=50)

# edge cases for aspect
if not aspect.strip():
    st.error("Please enter an aspect")
elif len(aspect.split(" "))>1:
    st.error("Please enter single aspect at max")



# button for predicting the sentiment
if st.button("Analyze Sentiment") and aspect!="" and review!="":
    print("akwejfkwebrkwrbwljrbqwb 3j3bt",review,aspect)

    # calling the predict function
    prediction=predict(review,aspect)
    st.write(f"The sentiment of the customer about the {aspect} is {prediction}")




