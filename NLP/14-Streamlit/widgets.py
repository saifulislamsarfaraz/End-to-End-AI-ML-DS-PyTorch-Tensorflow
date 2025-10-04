import streamlit as st
st.title("Streamlit Text Input")

name = st.text_input("Enter your name")

if name:
    st.write(f"Hello, {name}!")