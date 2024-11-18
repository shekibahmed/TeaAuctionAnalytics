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

        /* Analysis section styling */
        .analysis-section {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }

        /* Bullet point styling */
        .analysis-section ul {
            list-style-type: disc;
            margin-left: 1.5rem;
            padding-left: 1rem;
        }

        .analysis-section li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
            padding-left: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
