import streamlit as st

def apply_custom_styles():
    """Apply custom styling to the Streamlit app"""
    
    st.markdown("""
        <style>
        /* Base container styling */
        .main {
            padding: 2rem;
            max-width: 100%;
            box-sizing: border-box;
            overflow-x: hidden;
        }
        
        /* Viewport optimization */
        .stApp {
            overflow-x: hidden;
            width: 100vw;
            min-height: 100vh;
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
        
        /* Enhanced Tab styling with mobile optimization and layout containment */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.5rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: flex-start;
            min-height: 48px;
            width: 100%;
            box-sizing: border-box;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            contain: layout style paint;
            will-change: transform;
            transform: translateZ(0);
            backface-visibility: hidden;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: auto;
            min-height: 3rem;
            padding: 0.5rem 1rem;
            white-space: normal;
            background-color: #ffffff;
            border-radius: 4px;
            color: #1F4E79;
            font-weight: 500;
            flex: 0 0 auto;
            transition: transform 0.2s ease;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
            contain: content;
            transform: translateZ(0);
            will-change: transform;
        }
        
        /* Optimization for tab containers to reduce layout thrashing */
        .stTabs > div[role="tabpanel"] {
            contain: layout style;
            transform: translateZ(0);
            will-change: transform;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1F4E79 !important;
            color: white !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Mobile-specific optimizations */
        @media (max-width: 768px) {
            .stTabs [data-baseweb="tab-list"] {
                padding: 0.25rem;
                gap: 2px;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.35rem 0.75rem;
                font-size: 0.9rem;
            }
        }
        
        /* Collapsible section styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-weight: 600;
            color: #1F4E79;
        }
        
        .streamlit-expanderContent {
            border: 1px solid #e1e4e8;
            border-radius: 0 0 0.5rem 0.5rem;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
