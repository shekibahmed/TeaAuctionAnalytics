import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import openai
import os
import logging


def get_closest_match(column: str,
                      possible_matches: List[str]) -> Tuple[str, float]:
    """Find the closest matching column name based on string similarity"""
    from difflib import SequenceMatcher

    # Convert to lowercase for comparison
    column_lower = column.lower().strip()
    scores = [(match, SequenceMatcher(None, column_lower,
                                      match.lower().strip()).ratio())
              for match in possible_matches]
    return max(scores, key=lambda x: x[1])


def standardize_market_category(category: str) -> str:
    """Standardize market category name to the format: {region} CTC {type}"""
    # Remove extra spaces and standardize case
    category = ' '.join(category.strip().split())

    # Extract region and type
    if ' CTC ' in category:
        region, type_part = category.split(' CTC ')
    else:
        parts = category.split()
        if len(parts) < 3 or parts[0] not in [
                'North', 'South'
        ] or parts[1] != 'India' or parts[-1] not in ['Leaf', 'Dust']:
            return category  # Return original if format is completely invalid
        region = ' '.join(parts[:2])
        type_part = parts[-1]

    # Validate and standardize region
    if region not in ['North India', 'South India']:
        return category

    # Validate and standardize type
    if type_part not in ['Leaf', 'Dust']:
        return category

    # Return standardized format
    return f"{region} CTC {type_part}"


def process_excel_data(file):
    """Process uploaded Excel file and return formatted DataFrame"""
    # Column name variations mapping
    column_variations = {
        'Centre': [
            'Centre', 'Center', 'center', 'centre', 'Market Center',
            'Market Centre', 'Location'
        ],
        'Sale No':
        ['Sale No', 'Sale Number', 'Sale_No', 'SaleNo', 'Sale #', 'Sale'],
        'Sales Price(Kg)':
        ['Sales Price(Kg)', 'Price/Kg', 'Price (Kg)', 'Sales Price', 'Price'],
        'Sold Qty (Ton)': [
            'Sold Qty (Ton)', 'Sold Quantity', 'Sold_Qty', 'Sold Amount',
            'Quantity Sold'
        ],
        'Unsold Qty (Ton)': [
            'Unsold Qty (Ton)', 'Unsold Quantity', 'Unsold_Qty',
            'Unsold Amount', 'Quantity Unsold'
        ]
    }

    required_columns = list(column_variations.keys())
    df = None
    column_mapping = {}

    try:
        # Add debug logging
        logging.debug(f"Reading file: {file.name}")
        
        # Reset file pointer to beginning to ensure we read from the start
        file.seek(0)
        
        # Detect file type and use appropriate reader with error handling
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                logging.debug("Reading CSV file")
            except Exception as e:
                logging.error(f"Error reading CSV file: {str(e)}")
                # Try with different encoding if standard fails
                file.seek(0)
                df = pd.read_csv(file, encoding='latin1')
                logging.debug("Reading CSV file with latin1 encoding")
        elif file.name.endswith('.xls'):
            # For older .xls files
            try:
                df = pd.read_excel(file, engine='xlrd')
                logging.debug("Reading XLS file with xlrd engine")
            except Exception as e:
                logging.error(f"Error with xlrd engine: {str(e)}")
                # Try openpyxl as fallback
                file.seek(0)
                df = pd.read_excel(file, engine='openpyxl')
                logging.debug("Fallback to openpyxl for XLS file")
        elif file.name.endswith('.xlsx'):
            # For newer .xlsx files
            try:
                df = pd.read_excel(file, engine='openpyxl')
                logging.debug("Reading XLSX file with openpyxl engine")
            except Exception as e:
                logging.error(f"Error with openpyxl engine: {str(e)}")
                # Try a more resilient approach if available
                file.seek(0)
                df = pd.read_excel(file, engine='openpyxl', storage_options={'mode': 'rb'})
                logging.debug("Reading XLSX file with binary mode")
        else:
            raise ValueError(f"Unsupported file format. Please use .xlsx, .xls, or .csv files")
                
        if df is None or df.empty:
            raise ValueError("No data found in the file")
            
        logging.debug(f"Successfully read file with {len(df)} rows")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        raise ValueError(f"Error reading file: {str(e)}")
    missing_columns = []
    unmapped_columns = []

    for required_col in required_columns:
        variations = column_variations[required_col]
        found = False

        # First try exact matches
        for var in variations:
            if var in df.columns:
                column_mapping[var] = required_col
                found = True
                break

        # If no exact match, try fuzzy matching
        if not found:
            potential_matches = [(col, get_closest_match(col, variations))
                                 for col in df.columns
                                 if col not in column_mapping.keys()]
            best_match = max(
                potential_matches,
                key=lambda x: x[1][1]) if potential_matches else (None, (None,
                                                                         0))

            if best_match[1][1] > 0.8:  # Threshold for similarity
                column_mapping[best_match[0]] = required_col
                found = True
            else:
                missing_columns.append(required_col)

        if not found:
            missing_columns.append(required_col)

    # Check for unmapped columns
    unmapped_columns = [col for col in df.columns if col not in column_mapping]

    # Generate detailed error message if needed
    if missing_columns:
        error_msg = "\nMissing required columns:\n"
        for col in missing_columns:
            error_msg += f"- {col} (accepted variations: {', '.join(column_variations[col])})\n"
        if unmapped_columns:
            error_msg += "\nUnmapped columns in your file:\n"
            error_msg += ", ".join(unmapped_columns)
        raise ValueError(error_msg)

    # Rename columns to standard names
    df = df.rename(columns=column_mapping)

    # Clean and format data
    df = df[required_columns]
    df = df.dropna()

    # Convert columns to appropriate types
    df['Sale No'] = pd.to_numeric(df['Sale No'])
    df['Sales Price(Kg)'] = pd.to_numeric(df['Sales Price(Kg)'])
    df['Sold Qty (Ton)'] = pd.to_numeric(df['Sold Qty (Ton)'])
    df['Unsold Qty (Ton)'] = pd.to_numeric(df['Unsold Qty (Ton)'])

    # Standardize market categories
    df['Centre'] = df['Centre'].astype(str).apply(standardize_market_category)

    return df


def analyze_levels(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze current market position and absolute values"""
    insights = []
    centre_df = df[df['Centre'] == centre].copy()
    
    if centre_df.empty:
        return ["No data available for the selected market."]

    # Latest market position
    latest_sale = centre_df['Sale No'].max()
    latest_data_df = centre_df[centre_df['Sale No'] == latest_sale]
    if latest_data_df.empty:
        return ["No data available for the latest sale period."]
    
    latest_data = latest_data_df.iloc[0]

    # Price level analysis with weighted average
    weighted_avg_price = (
        centre_df['Sales Price(Kg)'] *
        centre_df['Sold Qty (Ton)']).sum() / centre_df['Sold Qty (Ton)'].sum()
    latest_price = latest_data['Sales Price(Kg)']
    price_percentile = (centre_df['Sales Price(Kg)']
                        <= latest_price).mean() * 100

    insights.append(f"Price Levels:")
    insights.append(
        f"  • Current price: ₹{latest_price:.2f}/Kg (Sale {latest_sale})")
    insights.append(
        f"  • Historical weighted average: ₹{weighted_avg_price:.2f}/Kg")
    insights.append(
        f"  • Current price is at the {price_percentile:.1f}th percentile of historical prices"
    )

    # Volume level analysis
    total_qty = latest_data['Sold Qty (Ton)'] + latest_data['Unsold Qty (Ton)']
    avg_total_qty = (centre_df['Sold Qty (Ton)'] +
                     centre_df['Unsold Qty (Ton)']).mean()

    insights.append(f"\nVolume Levels:")
    insights.append(
        f"  • Current offering: {total_qty:,.0f} tons (Sale {latest_sale})")
    insights.append(
        f"  • Historical average offering: {avg_total_qty:,.0f} tons")
    volume_status = "above" if total_qty > avg_total_qty else "below"
    insights.append(
        f"  • Current volume is {abs(total_qty - avg_total_qty):,.0f} tons {volume_status} average"
    )

    return insights


def analyze_trends(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze time-series patterns and changes"""
    insights = []
    centre_df = df[df['Centre'] == centre].copy()
    
    if centre_df.empty:
        return ["No data available for the selected market."]
    
    # Sort by Sale No - handle sorting with ascending parameter to fix error
    centre_df = centre_df.sort_values(by='Sale No', ascending=True)

    # Calculate weighted average prices for trend analysis
    centre_df['Weighted_Price'] = (
        centre_df['Sales Price(Kg)'] *
        centre_df['Sold Qty (Ton)']) / centre_df['Sold Qty (Ton)']
    price_changes = centre_df['Weighted_Price'].pct_change()
    recent_price_trend = price_changes.tail(3).mean()

    insights.append(f"Price Trends:")
    if abs(recent_price_trend) < 0.01:
        insights.append(
            "  • Weighted average prices have remained stable in recent sales")
    else:
        trend_direction = "upward" if recent_price_trend > 0 else "downward"
        insights.append(
            f"  • Recent {trend_direction} weighted price trend of {abs(recent_price_trend)*100:.1f}%"
        )

    # Volume trends
    volume_changes = (centre_df['Sold Qty (Ton)'] +
                      centre_df['Unsold Qty (Ton)']).pct_change()
    recent_volume_trend = volume_changes.tail(3).mean()

    insights.append(f"\nVolume Trends:")
    if abs(recent_volume_trend) < 0.05:
        insights.append("  • Offering volumes have been stable")
    else:
        trend_direction = "increasing" if recent_volume_trend > 0 else "decreasing"
        insights.append(
            f"  • Offering volumes are {trend_direction} by {abs(recent_volume_trend)*100:.1f}%"
        )

    # Market efficiency trends
    efficiency_ratio = centre_df['Sold Qty (Ton)'] / (
        centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
    recent_efficiency_trend = efficiency_ratio.tail(3).mean()

    insights.append(f"\nEfficiency Trends:")
    insights.append(
        f"  • Recent market efficiency: {recent_efficiency_trend*100:.1f}%")
    if recent_efficiency_trend > 0.8:
        insights.append("  • Market showing strong absorption capacity")
    elif recent_efficiency_trend < 0.6:
        insights.append("  • Market showing weak absorption capacity")

    return insights


def analyze_comparatives(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze cross-market comparisons and benchmarking with optimized data handling for large datasets"""
    insights = []
    
    try:
        # Handle empty dataframe
        if df.empty:
            return ["No data available for analysis."]
            
        # Extract region and tea type from centre name
        if ' CTC ' not in centre:
            return ["Invalid market format. Expected format: {region} CTC {type}"]
            
        region, tea_type = centre.split(' CTC ')
        tea_type = tea_type.strip() if tea_type else ""  # Clean any whitespace
        
        # Filter markets by region and tea type for fair comparison
        filtered_markets = [
            market for market in df['Centre'].unique()
            if market.split(' CTC ')[0] == region and 
            market.split(' CTC ')[1].strip() == tea_type
        ]
        
        # Add debug logging
        logging.debug(f"Analyzing market: {centre}")
        logging.debug(f"Region: {region}, Tea Type: {tea_type}")
        logging.debug(f"Filtered markets: {filtered_markets}")
        
        # Calculate metrics for current centre
        current_data = df[df['Centre'] == centre].copy()
        if current_data.empty:
            logging.warning(f"No data found for {centre}")
            return [f"No data available for {centre}"]
            
        # Calculate weighted average price for current market
        current_price = calculate_weighted_price(current_data)
        logging.debug(f"Calculated price for {centre}: {current_price}")
        
        # Calculate recent market efficiency
        total_volume = current_data['Sold Qty (Ton)'] + current_data['Unsold Qty (Ton)']
        current_efficiency = (current_data['Sold Qty (Ton)'] / total_volume).mean()
        
        insights.append(f"Market Comparatives Analysis for {centre}:")
        insights.append(f"• Current market weighted average price: ₹{current_price:.2f}/Kg")
        insights.append(f"• Market efficiency: {current_efficiency*100:.1f}%")
        
        # Compare with other markets of same tea type
        price_differences = []
        efficiency_differences = []
        
        for other_market in filtered_markets:
            if other_market != centre:
                other_data = df[df['Centre'] == other_market].copy()
                if not other_data.empty:
                    # Calculate price metrics
                    other_price = calculate_weighted_price(other_data)
                    price_diff = current_price - other_price
                    price_ratio = current_price / other_price if other_price > 0 else 0
                    
                    # Calculate efficiency metrics
                    other_total_volume = other_data['Sold Qty (Ton)'] + other_data['Unsold Qty (Ton)']
                    other_efficiency = (other_data['Sold Qty (Ton)'] / other_total_volume).mean()
                    
                    price_differences.append(price_diff)
                    efficiency_differences.append(current_efficiency - other_efficiency)
                    
                    insights.append(f"\nComparison with {other_market}:")
                    insights.append(f"• Price difference: {price_diff:+.2f} ₹/Kg")
                    insights.append(f"• Price ratio: {price_ratio:.2f}")
                    insights.append(f"• Efficiency difference: {(current_efficiency - other_efficiency)*100:+.1f}%")
        
        # Add market position summary
        if price_differences:
            avg_price_diff = sum(price_differences) / len(price_differences)
            avg_efficiency_diff = sum(efficiency_differences) / len(efficiency_differences)
            
            insights.append(f"\nMarket Position Summary:")
            insights.append(f"• Average price difference: {avg_price_diff:+.2f} ₹/Kg vs other {tea_type} markets")
            insights.append(f"• Average efficiency difference: {avg_efficiency_diff*100:+.1f}% vs other {tea_type} markets")
        
        return insights
        
    except Exception as e:
        logging.error(f"Error in comparative analysis: {str(e)}")
        return [f"Error in comparative analysis: {str(e)}"]

def calculate_weighted_price(data: pd.DataFrame | pd.Series) -> float:
    """Calculate weighted average price for a given dataset using batch processing"""
    batch_size = 1000  # Define batch size for processing
    
    # Ensure we have a DataFrame
    if isinstance(data, pd.Series):
        return 0
        
    if len(data) == 0:
        return 0

    try:
        total_weighted_sum = 0
        total_weight = 0

        # Process data in batches to handle large datasets efficiently
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]

            # Convert to numpy arrays for faster computation
            weights = batch['Sold Qty (Ton)'].to_numpy()
            prices = batch['Sales Price(Kg)'].to_numpy()

            # Filter out any NaN values
            mask = np.isfinite(weights) & np.isfinite(prices)
            valid_weights = weights[mask]
            valid_prices = prices[mask]

            # Calculate batch totals
            batch_weighted_sum = np.sum(valid_weights * valid_prices)
            batch_weight = np.sum(valid_weights)

            total_weighted_sum += batch_weighted_sum
            total_weight += batch_weight

        logging.debug(f"Processed {len(data)} records in {len(data)//batch_size + 1} batches")
        logging.debug(f"Total weight processed: {total_weight:.2f} tons")

        if total_weight > 0:
            result = total_weighted_sum / total_weight
            logging.debug(f"Calculated weighted price: ₹{result:.2f}/Kg")
            return result
        return 0

    except Exception as e:
        logging.error(f"Error in calculate_weighted_price: {str(e)}")
        return 0
            

    


def generate_ai_narrative(df: pd.DataFrame, centre: str) -> str:
    """Generate AI-powered narrative market analysis"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return "AI narrative generation unavailable: OpenAI API key not found in environment variables."

        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Check if data is available
        filtered_df = df[df['Centre'] == centre].copy()
        if filtered_df.empty:
            return "No data available for the selected market."
            
        # Sort by Sale No with proper syntax
        centre_df = filtered_df.sort_values(by='Sale No', ascending=True)

        # Calculate comprehensive metrics
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        weighted_avg_price = (centre_df['Sales Price(Kg)'] *
                              centre_df['Sold Qty (Ton)']
                              ).sum() / centre_df['Sold Qty (Ton)'].sum()
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        price_volatility = centre_df['Sales Price(Kg)'].std()

        # Volume metrics
        total_volume = centre_df['Sold Qty (Ton)'] + centre_df[
            'Unsold Qty (Ton)']
        volume_trend = total_volume.pct_change().mean()

        # Efficiency metrics
        efficiency_ratio = centre_df['Sold Qty (Ton)'] / total_volume
        recent_efficiency = efficiency_ratio.tail(3).mean()

        market_context = f"""
        Market: {centre}
        Current Price: ₹{current_price:.2f}/Kg
        Weighted Average Price: ₹{weighted_avg_price:.2f}/Kg
        Price Trend: {price_trend*100:.1f}% average change
        Price Volatility: ₹{price_volatility:.2f}/Kg
        Volume Trend: {volume_trend*100:.1f}% average change
        Market Efficiency: {recent_efficiency*100:.1f}%
        """

        prompt = f"""You are a tea market analyst. Analyze the following tea market data and provide 3-4 detailed sentences covering:
1. Current price trends and market position
2. Volume dynamics and market efficiency
3. Future market outlook based on current indicators
4. Key factors influencing market behavior

Market Data:
{market_context}"""

        response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }],
                                                  temperature=0.7,
                                                  max_tokens=300)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in AI narrative generation: {str(e)}")
        return f"AI narrative generation encountered an error: {str(e)}"


def generate_price_analysis(df: pd.DataFrame, centre: str) -> str:
    """Generate detailed price analysis including trends, seasonality, and forecasts"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return "Price analysis unavailable: OpenAI API key not found in environment variables."

        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Check if data is available
        filtered_df = df[df['Centre'] == centre].copy()
        if filtered_df.empty:
            return "No data available for the selected market."
            
        # Sort by Sale No with proper syntax
        centre_df = filtered_df.sort_values(by='Sale No', ascending=True)

        # Calculate comprehensive price metrics
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        weighted_avg_price = (centre_df['Sales Price(Kg)'] *
                              centre_df['Sold Qty (Ton)']
                              ).sum() / centre_df['Sold Qty (Ton)'].sum()
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        price_volatility = centre_df['Sales Price(Kg)'].std()
        price_range = centre_df['Sales Price(Kg)'].max(
        ) - centre_df['Sales Price(Kg)'].min()
        price_percentile = (centre_df['Sales Price(Kg)']
                            <= current_price).mean() * 100

        # Calculate seasonal patterns (if enough data)
        try:
            # Convert Sale No to datetime for seasonal analysis
            centre_df['Date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
                centre_df['Sale No'] * 7, unit='D')
            sales_by_month = centre_df.groupby(pd.Grouper(
                key='Date', freq='M'))['Sales Price(Kg)'].mean()
            seasonal_pattern = "Detected" if sales_by_month.std(
            ) > price_volatility * 0.5 else "Not significant"
        except Exception as e:
            seasonal_pattern = "Analysis not available"
            logging.error(f"Error in seasonal analysis: {str(e)}")

        market_context = f"""
        Market: {centre}
        Current Price: ₹{current_price:.2f}/Kg
        Weighted Average Price: ₹{weighted_avg_price:.2f}/Kg
        Price Trend: {price_trend*100:.1f}% average change
        Price Volatility: ₹{price_volatility:.2f}/Kg
        Price Range: ₹{price_range:.2f}/Kg
        Current Price Percentile: {price_percentile:.1f}%
        Seasonal Pattern: {seasonal_pattern}
        """

        prompt = f"""You are a tea market price analyst. Analyze the following price data and provide 4-5 detailed insights focusing on:
1. Current price position and historical context
2. Price volatility and market stability
3. Seasonal patterns and cyclical behavior
4. Short-term price forecast
5. Comparative price analysis

Present insights as complete sentences without any formatting.

Market Data:
{market_context}"""

        response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }],
                                                  temperature=0.7,
                                                  max_tokens=400)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in price analysis: {str(e)}")
        return "Price analysis currently unavailable. Please try again later."


def generate_market_insights(df: pd.DataFrame, centre: str) -> str:
    """Generate market position and competitive analysis insights"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return "Market insights unavailable: OpenAI API key not found in environment variables."

        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Check if data is available
        filtered_df = df[df['Centre'] == centre].copy()
        if filtered_df.empty:
            return "No data available for the selected market."
            
        # Sort by Sale No with proper syntax
        centre_df = filtered_df.sort_values(by='Sale No', ascending=True)

        # Calculate comprehensive market metrics
        market_share = centre_df['Sold Qty (Ton)'].sum(
        ) / df['Sold Qty (Ton)'].sum()
        efficiency = centre_df['Sold Qty (Ton)'].sum() / (
            centre_df['Sold Qty (Ton)'].sum() +
            centre_df['Unsold Qty (Ton)'].sum())

        # Additional metrics for competitive analysis
        avg_lot_size = (centre_df['Sold Qty (Ton)'] +
                        centre_df['Unsold Qty (Ton)']).mean()
        sales_growth = centre_df['Sold Qty (Ton)'].pct_change().mean()
        market_stability = 1 - (centre_df['Unsold Qty (Ton)'] /
                                (centre_df['Sold Qty (Ton)'] +
                                 centre_df['Unsold Qty (Ton)'])).std()

        market_context = f"""
        Market: {centre}
        Market Share: {market_share*100:.1f}%
        Market Efficiency: {efficiency*100:.1f}%
        Average Lot Size: {avg_lot_size:.1f} tons
        Sales Growth: {sales_growth*100:.1f}%
        Market Stability: {market_stability*100:.1f}%
        """

        prompt = f"""You are a tea market analyst. Analyze the following market data and provide 3-4 detailed insights focusing on:
1. Market position and competitive strength
2. Operational efficiency and market absorption
3. Growth trends and market dynamics
4. Strategic recommendations

Present insights as complete sentences without any formatting.

Market Data:
{market_context}"""

        response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }],
                                                  temperature=0.7,
                                                  max_tokens=300)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in market insights: {str(e)}")
        return "Market insights currently unavailable. Please try again later."


def generate_pdf_report(df: pd.DataFrame, centre: str) -> bytes:
    """Generate a PDF report with statistical analysis for the selected market"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO

    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle('CustomTitle',
                                 parent=styles['Heading1'],
                                 fontSize=24,
                                 spaceAfter=30)

    heading_style = ParagraphStyle('CustomHeading',
                                   parent=styles['Heading2'],
                                   fontSize=16,
                                   spaceAfter=20)

    normal_style = ParagraphStyle('CustomNormal',
                                  parent=styles['Normal'],
                                  fontSize=12,
                                  spaceAfter=12)

    # Add content
    story.append(Paragraph(f"Market Analysis Report - {centre}", title_style))
    story.append(Spacer(1, 20))

    # Add Statistical Analysis
    for section, content in [
        ("Price and Volume Levels", analyze_levels(df, centre)),
        ("Market Trends", analyze_trends(df, centre)),
        ("Market Comparatives", analyze_comparatives(df, centre))
    ]:
        story.append(Paragraph(section, heading_style))
        for insight in content:
            story.append(Paragraph(insight.replace('₹', 'Rs.'), normal_style))
        story.append(Spacer(1, 20))

    # Generate PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def calculate_correlations(df: pd.DataFrame, centre: str) -> pd.DataFrame:
    """Calculate correlations between different metrics for a given centre"""
    centre_df = df[df['Centre'] == centre].copy()

    # Calculate additional metrics
    centre_df['Total Volume'] = centre_df['Sold Qty (Ton)'] + centre_df[
        'Unsold Qty (Ton)']
    centre_df['Market Efficiency'] = centre_df['Sold Qty (Ton)'] / centre_df[
        'Total Volume']
    centre_df['Price Change'] = centre_df['Sales Price(Kg)'].pct_change()
    centre_df['Volume Change'] = centre_df['Total Volume'].pct_change()

    # Select metrics for correlation
    correlation_metrics = [
        'Sales Price(Kg)', 'Total Volume', 'Market Efficiency', 'Price Change',
        'Volume Change', 'Sold Qty (Ton)', 'Unsold Qty (Ton)'
    ]

    # Calculate correlation matrix
    correlation_matrix = centre_df[correlation_metrics].corr()

    return correlation_matrix


def analyze_key_correlations(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze and explain key correlations between metrics"""
    correlation_matrix = calculate_correlations(df, centre)
    insights = []

    # Analyze price correlations
    price_vol_corr = correlation_matrix.loc['Sales Price(Kg)', 'Total Volume']
    price_eff_corr = correlation_matrix.loc['Sales Price(Kg)',
                                            'Market Efficiency']

    insights.append("Price Correlations:")
    insights.append(f"• Price-Volume Correlation: {price_vol_corr:.2f}")
    if abs(price_vol_corr) > 0.5:
        direction = "positive" if price_vol_corr > 0 else "negative"
        insights.append(
            f"  - Strong {direction} relationship between price and volume")

    insights.append(f"• Price-Efficiency Correlation: {price_eff_corr:.2f}")
    if abs(price_eff_corr) > 0.5:
        direction = "positive" if price_eff_corr > 0 else "negative"
        insights.append(
            f"  - Strong {direction} relationship between price and market efficiency"
        )

    # Analyze volume correlations
    vol_eff_corr = correlation_matrix.loc['Total Volume', 'Market Efficiency']
    insights.append("\nVolume Correlations:")
    insights.append(f"• Volume-Efficiency Correlation: {vol_eff_corr:.2f}")
    if abs(vol_eff_corr) > 0.5:
        direction = "positive" if vol_eff_corr > 0 else "negative"
        insights.append(
            f"  - Strong {direction} relationship between volume and market efficiency"
        )

    return insights


if __name__ == "__main__":
    # Example usage
    df = process_excel_data('your_file.xlsx')
    centre = 'North India CTC Leaf'  # Example market
    print(generate_price_analysis(df, centre))
    # Add more analysis calls as needed
    # print(generate_ai_narrative(df, centre))
    # print(generate_market_insights(df, centre))
    # print(generate_pdf_report(df, centre))
