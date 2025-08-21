import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/query"  # Update this if your FastAPI app runs on a different host/port

# Set page configuration
st.set_page_config(
    page_title="LLM Query Prototype",
    page_icon=":)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for navigation and settings
st.sidebar.title("LLM Query Prototype")
st.sidebar.markdown(
    """
    This app allows you to interact with an advanced AI assistant powered by Llama.
    You can ask questions and optionally provide context for more accurate responses.
    """
)
st.sidebar.image("https://via.placeholder.com/150", caption="AI Assistant", use_column_width=True)

# Main app layout
st.title("🤖 LLM Query Interface")
st.markdown(
    """
    Welcome to the **LLM Query Interface**! Use the form below to interact with the AI assistant.
    """
)

# Input fields
st.subheader("Ask a Question")
question = st.text_input("Enter your question:", placeholder="Type your question here...")
context = st.text_area("Optional Context", placeholder="Provide additional context if needed...")

# Submit button
if st.button("Submit Query"):
    if not question.strip():
        st.error("Please enter a valid question.")
    else:
        # Send request to FastAPI backend
        try:
            with st.spinner("Processing your query..."):
                response = requests.get(API_URL, params={"question": question, "context": context})
                if response.status_code == 200:
                    result = response.json()
                    st.success("Response from Llama:")
                    st.write(result["response"])
                else:
                    st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    **Note**: This is a prototype application. For production use, ensure proper error handling, security, and scalability.
    """
)