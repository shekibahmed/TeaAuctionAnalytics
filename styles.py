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

        /* Analysis sections styling */
        .analysis-section {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
        }

        /* Content container for better alignment */
        .analysis-content {
            margin: 0.5rem 0;
            padding: 0 1rem;
            line-height: 1.6;
        }

        /* General bullet point styling */
        .analysis-content ul {
            list-style-type: disc !important;
            margin: 0.75rem 0 !important;
            padding-left: 2rem !important;
        }

        .analysis-content li {
            margin: 0.5rem 0 !important;
            padding-left: 0.5rem !important;
            line-height: 1.5 !important;
            display: list-item !important;
            text-align: left !important;
        }

        /* Handle nested lists */
        .analysis-content ul ul {
            margin: 0.5rem 0 0.5rem 1rem !important;
            list-style-type: circle !important;
        }

        /* Custom spacing for different analysis sections */
        #market-narrative .analysis-content,
        #price-analysis .analysis-content,
        #market-insights .analysis-content {
            margin-bottom: 1rem;
        }

        /* Fix for streamlit markdown container */
        div[data-testid="stMarkdownContainer"] ul {
            padding-left: 2rem !important;
            margin: 0.75rem 0 !important;
        }

        div[data-testid="stMarkdownContainer"] li {
            margin: 0.5rem 0 !important;
            padding-left: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
