import streamlit as st

def apply_custom_styles():
    """Apply custom styling to the Streamlit app"""
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        
        .stTitle {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1F4E79;
            margin-bottom: 2rem;
        }
        
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stMetric:hover {
            transform: translateY(-2px);
            transition: all 0.2s ease;
        }
        
        .stDataFrame {
            border: 1px solid #e1e4e8;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        .stSubheader {
            color: #1F4E79;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
        }
        
        .stAlert {
            border-radius: 0.5rem;
            border: none;
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #1F4E79;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #1F4E79;
            color: white;
            border-radius: 0.5rem;
        }
        
        .stButton>button:hover {
            background-color: #153557;
            border-color: #153557;
        }
        </style>
    """, unsafe_allow_html=True)
