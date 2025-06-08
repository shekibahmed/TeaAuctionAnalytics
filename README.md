
# CTC Tea Sales Analytics Dashboard

A comprehensive Streamlit-powered analytics dashboard for analyzing CTC (Crush, Tear, Curl) tea sales data across North and South India markets. This application provides AI-powered market insights, statistical analysis, and interactive visualizations for tea auction data.

## Features

### 📊 Core Analytics
- **Interactive Data Visualization**: Dynamic charts showing price trends, volume analysis, and market efficiency
- **Market Comparison**: Compare different tea markets (North/South India, Leaf/Dust varieties)
- **Statistical Analysis**: Comprehensive analysis including correlations, trends, and market positions
- **Data Processing**: Support for Excel (.xlsx, .xls) and CSV file formats with intelligent column mapping

### 🤖 AI-Powered Insights
- **Market Narrative**: AI-generated market analysis and insights
- **Price Analysis**: Detailed price trend analysis with forecasting
- **Market Intelligence**: Competitive analysis and strategic recommendations
- **Automated Reporting**: AI-powered insights generation using OpenAI GPT models

### 📈 Advanced Features
- **Multi-tab Interface**: Organized analysis across Position, Trends, Comparison, and Levels tabs
- **Responsive Design**: Mobile-optimized interface with touch-friendly controls
- **Real-time Processing**: Dynamic data filtering and analysis
- **Export Capabilities**: PDF report generation for detailed analysis

### 🫖 Tea-Themed Loading Animations
- **Welcome Brewing Sequence**: Animated introduction for first-time users with tea preparation stages
- **Data Upload Animations**: Tea steeping progress indicators during file processing
- **AI Analysis Animations**: Tea master tasting animations for market intelligence generation
- **Visualization Brewing**: Tea preparation animations while charts are being generated
- **Correlation Blending**: Tea blending animations for relationship analysis
- **Progress Tracking**: Multi-stage tea ceremony progress indicators with elapsed time

## Technology Stack

- **Frontend**: Streamlit 1.29.0
- **Data Processing**: Pandas 2.1.3, NumPy 1.26.2
- **Visualization**: Plotly 5.18.0, Plotly Express
- **AI Integration**: OpenAI 1.3.5
- **File Processing**: openpyxl, xlrd
- **Report Generation**: ReportLab
- **Python**: 3.11+

## Installation

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ctc-tea-sales-analytics.git
   cd ctc-tea-sales-analytics
   ```

2. **Install dependencies using pip**:
   ```bash
   pip install .
   ```
   
   Or install from pyproject.toml directly:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables** (optional for AI features):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

### Alternative Installation Methods

**Using pip with dependencies list**:
```bash
pip install streamlit==1.29.0 pandas==2.1.3 plotly==5.18.0 openai==1.3.5 numpy==1.26.2 openpyxl xlrd reportlab plotly-express
```

**For development**:
```bash
pip install -e ".[dev]"
```

## Usage

### Running the Application

**Local Development**:
```bash
streamlit run main.py --server.port=5000 --server.address=0.0.0.0
```

**On Replit**:
Click the "Run" button or use the configured workflow.

### Data Format

The application expects Excel or CSV files with the following columns (accepts variations):

| Required Column | Accepted Variations |
|----------------|-------------------|
| Centre | Centre, Center, Market Center, Location |
| Sale No | Sale No, Sale Number, Sale_No, SaleNo |
| Sales Price(Kg) | Price/Kg, Price (Kg), Sales Price, Price |
| Sold Qty (Ton) | Sold Quantity, Sold_Qty, Quantity Sold |
| Unsold Qty (Ton) | Unsold Quantity, Unsold_Qty, Quantity Unsold |

### Market Categories

Markets should follow the format: `{Region} CTC {Type}`
- **Regions**: North India, South India
- **Types**: Leaf, Dust

Example: "North India CTC Leaf", "South India CTC Dust"

## Features Overview

### 1. Data Upload & Processing
- Drag-and-drop file upload
- Intelligent column mapping with fuzzy matching
- Data validation and cleaning
- Support for multiple file formats

### 2. Market Selection
- Multi-select filters for regions and tea types
- Dynamic market filtering
- Real-time data updates

### 3. Analysis Tabs

#### 📊 Position Analysis
- Current market position metrics
- Price, volume, and efficiency analysis
- Historical context and percentile rankings

#### 📈 Trends Analysis
- Price trend visualization with trend lines
- Market efficiency tracking
- Rolling correlations and moving averages

#### 🔄 Comparative Analysis
- Cross-market comparisons
- Correlation heatmaps
- Competitive benchmarking

#### 💰 Levels Analysis
- Price distribution analysis
- Volume analysis with multiple aggregation methods
- Statistical summaries and insights

### 4. AI-Powered Insights
- **Market Narrative**: Comprehensive market analysis
- **Price Analysis**: Detailed price insights and forecasting
- **Market Intelligence**: Strategic recommendations

## File Structure

```
├── main.py              # Main Streamlit application
├── utils.py             # Data processing and analysis utilities
├── styles.py            # Custom CSS styling
├── loading_animations.py # Tea-themed loading animation system
├── assets/              # Sample data and resources
│   └── default_data.csv # Sample tea market dataset
├── pyproject.toml       # Project dependencies
├── .replit              # Replit configuration
└── README.md            # This file
```

## Animation System (`loading_animations.py`)

The dashboard features a comprehensive tea-themed loading animation system that enhances user experience during data processing and analysis operations.

### Animation Categories

#### 🫖 **TeaLoadingAnimations** - Core Animation Class
- **Tea Messages**: 50+ themed messages across 5 categories (upload, processing, ai_analysis, visualization, report)
- **Tea Icons**: 10 animated tea-related emojis (teapot, cup, leaves, steam, etc.)
- **Progress Animations**: Spinning teapot, tea leaf progress bars, steaming cup sequences

#### 🎭 **ProgressTracker** - Context Manager
- **Multi-step Progress**: Tracks progress across multiple processing stages
- **Elapsed Time Display**: Shows processing duration for transparency
- **Automatic Cleanup**: Removes animation placeholders when complete
- **Category-based Messages**: Different message sets for different operations

#### 🏭 **TeaStationAnimator** - Advanced Animations
- **Chart Brewing**: 4-stage visualization preparation with progress tracking
- **Correlation Blending**: Tea blending metaphor for relationship analysis
- **Market Intelligence**: Sophisticated tea tasting animation for AI analysis

#### 🎪 **EnhancedProgressTracker** - Ceremony Themes
- **Traditional Ceremony**: Classical tea preparation with reverent styling
- **Modern Station**: Contemporary brewing with digital aesthetics
- **Artisan Crafting**: Hand-crafted approach with premium styling

### Animation Integration Points

1. **App Initialization**: Welcome brewing sequence on first visit
2. **Data Upload**: Tea steeping animations during file processing
3. **Sample Data Loading**: Quick brewing animation for instant data access
4. **AI Analysis**: Tea master tasting sequence for all AI-powered insights
5. **Chart Generation**: Visualization brewing for market position charts
6. **Correlation Analysis**: Tea blending animation for relationship matrices
7. **Statistical Processing**: Multi-stage progress tracking for complex calculations

### Technical Implementation

- **HTML/CSS Animations**: Custom styled progress bars with gradient effects
- **Dynamic Content**: Real-time message updates with contextual tea terminology
- **Responsive Design**: Animations adapt to mobile and desktop viewports
- **Performance Optimized**: Lightweight animations that don't impact data processing
- **State Management**: Proper cleanup prevents UI conflicts and memory leaks

## Key Functions

### Data Processing (`utils.py`)
- `process_excel_data()`: File processing and column mapping
- `standardize_market_category()`: Market name standardization
- `calculate_weighted_price()`: Weighted average price calculations

### Analysis Functions
- `analyze_levels()`: Market position analysis
- `analyze_trends()`: Trend analysis and patterns
- `analyze_comparatives()`: Cross-market comparisons
- `calculate_correlations()`: Correlation matrix generation

### AI Functions
- `generate_ai_narrative()`: AI-powered market insights
- `generate_price_analysis()`: Price analysis with AI
- `generate_market_insights()`: Strategic market intelligence

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI-powered features

### Streamlit Configuration
The app includes custom styling and mobile optimization. Configuration can be found in `styles.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Deployment

### Replit Deployment
This application is optimized for Replit deployment:
- Click the "Deploy" button in Replit
- The app will be available on your Replit domain
- Supports both development and production modes

### Manual Deployment
For other platforms, ensure:
- Python 3.11+ environment
- All dependencies installed
- Environment variables configured
- Port 5000 accessible

## Troubleshooting

### Common Issues

1. **File Upload Errors**: Ensure your file has the required columns with accepted naming variations
2. **AI Features Not Working**: Check that `OPENAI_API_KEY` is properly set
3. **Performance Issues**: For large datasets, the app includes optimized batch processing

### Debug Mode
Run with debug logging:
```bash
streamlit run main.py --logger.level=debug
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the console logs for error details
3. Ensure all dependencies are properly installed

## Acknowledgments

- Built with Streamlit for the web interface
- Plotly for interactive visualizations
- OpenAI for AI-powered insights
- Pandas and NumPy for data processing

---

**Note**: This application is designed for tea market analysis and can be adapted for other commodity markets with similar data structures.
