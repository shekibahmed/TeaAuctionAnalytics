# CTC Tea Sales Analytics Dashboard

## Overview

The CTC Tea Sales Analytics Dashboard is a Streamlit-powered web application designed for analyzing Crush, Tear, Curl (CTC) tea sales data across North and South India markets. The application provides comprehensive market intelligence through AI-powered insights, interactive visualizations, and statistical analysis capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with a clean separation of concerns:

- **Frontend**: Streamlit-based web interface with custom styling and tea-themed animations
- **Data Processing**: Pandas-based data pipeline with intelligent column mapping and fuzzy matching
- **Visualization**: Plotly-powered interactive charts and graphs
- **AI Integration**: OpenAI GPT integration for market narrative generation
- **Session Management**: Streamlit session state for maintaining user data across interactions

## Key Components

### Core Application (`main.py`)
- Entry point for the Streamlit application
- Handles page configuration and initial setup
- Manages the main user interface flow
- Integrates all other modules

### Data Processing (`utils.py`)
- Excel and CSV file processing capabilities
- Intelligent column mapping with fuzzy string matching
- Market category standardization for North/South India CTC tea markets
- Statistical analysis functions including correlations and trends
- AI-powered market analysis integration

### User Interface Styling (`styles.py`)
- Custom CSS styling for the Streamlit application
- Responsive design optimization
- Mobile-friendly interface elements
- Tea-themed visual enhancements

### Loading Animations (`loading_animations.py`)
- Tea-themed loading animations and progress indicators
- Multi-stage progress tracking for different operations
- Context-aware animation messages for upload, processing, AI analysis, visualization, and reporting
- Enhanced user experience during data processing

## Data Flow

1. **File Upload**: Users upload Excel (.xlsx, .xls) or CSV files through the Streamlit interface
2. **Data Processing**: Files are processed using pandas with intelligent column mapping
3. **Market Standardization**: Tea market categories are standardized to consistent format
4. **Analysis Generation**: Statistical analysis, correlations, and trends are calculated
5. **AI Insights**: OpenAI GPT generates market narratives and strategic insights
6. **Visualization**: Interactive charts and graphs are created using Plotly
7. **Results Display**: Multi-tab interface presents analysis across Position, Trends, Comparison, and Levels

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations

### AI Integration
- **OpenAI**: GPT integration for market intelligence generation
- API key required as environment variable (`OPENAI_API_KEY`)

### Data Processing
- **openpyxl**: Excel file handling
- **difflib**: Fuzzy string matching for column mapping

## Deployment Strategy

The application is designed for deployment on Replit with the following considerations:

### Environment Setup
- Python 3.11 or higher required
- Dependencies managed through pip installation
- Environment variables for API key configuration

### Replit Configuration
- Main entry point: `main.py`
- Default port: 5000
- Server address: 0.0.0.0 for external access

### Security Considerations
- Client-side data processing (no persistent storage)
- Environment variable management for API keys
- File upload restrictions (Excel and CSV only)
- Session-based data management with automatic cleanup

### Performance Optimization
- Batch processing for large datasets
- Memory management during file processing
- Progressive loading with tea-themed animations
- Responsive design for various screen sizes

The application emphasizes user experience through tea-themed animations, comprehensive market analysis capabilities, and AI-powered insights while maintaining security and performance standards suitable for production deployment.