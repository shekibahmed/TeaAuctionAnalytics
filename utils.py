import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import openai
import os
import logging

def get_closest_match(column: str, possible_matches: List[str]) -> Tuple[str, float]:
    """Find the closest matching column name based on string similarity"""
    from difflib import SequenceMatcher
    
    # Convert to lowercase for comparison
    column_lower = column.lower().strip()
    scores = [(match, SequenceMatcher(None, column_lower, match.lower().strip()).ratio())
             for match in possible_matches]
    return max(scores, key=lambda x: x[1])

def process_excel_data(file):
    """Process uploaded Excel file and return formatted DataFrame"""
    # Column name variations mapping
    column_variations = {
        'Centre': ['Centre', 'Center', 'center', 'centre', 'Market Center', 'Market Centre', 'Location'],
        'Sale No': ['Sale No', 'Sale Number', 'Sale_No', 'SaleNo', 'Sale #', 'Sale'],
        'Sales Price(Kg)': ['Sales Price(Kg)', 'Price/Kg', 'Price (Kg)', 'Sales Price', 'Price'],
        'Sold Qty (Ton)': ['Sold Qty (Ton)', 'Sold Quantity', 'Sold_Qty', 'Sold Amount', 'Quantity Sold'],
        'Unsold Qty (Ton)': ['Unsold Qty (Ton)', 'Unsold Quantity', 'Unsold_Qty', 'Unsold Amount', 'Quantity Unsold']
    }
    
    required_columns = list(column_variations.keys())
    
    # Read the file
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")
    
    # Print actual columns for debugging
    print("\nActual columns in uploaded file:", df.columns.tolist())
    
    # Map columns to standard names
    column_mapping = {}
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
                               for col in df.columns if col not in column_mapping.keys()]
            best_match = max(potential_matches, key=lambda x: x[1][1]) if potential_matches else (None, (None, 0))
            
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
    
    # Validate market categories
    valid_categories = [
        'North India CTC Leaf', 'North India CTC Dust',
        'South India CTC Leaf', 'South India CTC Dust'
    ]
    
    invalid_categories = set(df['Centre'].unique()) - set(valid_categories)
    if invalid_categories:
        raise ValueError(f"Invalid market categories found: {invalid_categories}. Valid categories are: " + ", ".join(valid_categories))
    
    return df

def analyze_levels(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze current market position and absolute values"""
    insights = []
    centre_df = df[df['Centre'] == centre].copy()
    
    # Latest market position
    latest_sale = centre_df['Sale No'].max()
    latest_data = centre_df[centre_df['Sale No'] == latest_sale].iloc[0]
    
    # Price level analysis
    avg_price = centre_df['Sales Price(Kg)'].mean()
    latest_price = latest_data['Sales Price(Kg)']
    price_percentile = (centre_df['Sales Price(Kg)'] <= latest_price).mean() * 100
    
    insights.append(f"Price Levels:")
    insights.append(f"  • Current price: ₹{latest_price:.2f}/Kg (Sale {latest_sale})")
    insights.append(f"  • Historical average: ₹{avg_price:.2f}/Kg")
    insights.append(f"  • Current price is at the {price_percentile:.1f}th percentile of historical prices")
    
    # Volume level analysis
    total_qty = latest_data['Sold Qty (Ton)'] + latest_data['Unsold Qty (Ton)']
    avg_total_qty = (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']).mean()
    
    insights.append(f"\nVolume Levels:")
    insights.append(f"  • Current offering: {total_qty:,.0f} tons (Sale {latest_sale})")
    insights.append(f"  • Historical average offering: {avg_total_qty:,.0f} tons")
    volume_status = "above" if total_qty > avg_total_qty else "below"
    insights.append(f"  • Current volume is {abs(total_qty - avg_total_qty):,.0f} tons {volume_status} average")
    
    return insights

def analyze_trends(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze time-series patterns and changes"""
    insights = []
    centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
    
    # Price trends
    price_changes = centre_df['Sales Price(Kg)'].pct_change()
    recent_price_trend = price_changes.tail(3).mean()
    
    insights.append(f"Price Trends:")
    if abs(recent_price_trend) < 0.01:
        insights.append("  • Prices have remained stable in recent sales")
    else:
        trend_direction = "upward" if recent_price_trend > 0 else "downward"
        insights.append(f"  • Recent {trend_direction} price trend of {abs(recent_price_trend)*100:.1f}%")
    
    # Volume trends
    volume_changes = (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']).pct_change()
    recent_volume_trend = volume_changes.tail(3).mean()
    
    insights.append(f"\nVolume Trends:")
    if abs(recent_volume_trend) < 0.05:
        insights.append("  • Offering volumes have been stable")
    else:
        trend_direction = "increasing" if recent_volume_trend > 0 else "decreasing"
        insights.append(f"  • Offering volumes are {trend_direction} by {abs(recent_volume_trend)*100:.1f}%")
    
    # Market efficiency trends
    efficiency_ratio = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
    recent_efficiency_trend = efficiency_ratio.tail(3).mean()
    
    insights.append(f"\nEfficiency Trends:")
    insights.append(f"  • Recent market efficiency: {recent_efficiency_trend*100:.1f}%")
    if recent_efficiency_trend > 0.8:
        insights.append("  • Market showing strong absorption capacity")
    elif recent_efficiency_trend < 0.6:
        insights.append("  • Market showing weak absorption capacity")
    
    return insights

def analyze_comparatives(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze cross-market comparisons and benchmarking"""
    insights = []
    region, tea_type = centre.split(' CTC ')
    
    # Compare with other type in same region
    other_type = 'Dust' if tea_type == 'Leaf' else 'Leaf'
    other_centre = f"{region} CTC {other_type}"
    
    if other_centre in df['Centre'].unique():
        centre_df = df[df['Centre'] == centre].copy()
        other_df = df[df['Centre'] == other_centre].copy()
        
        # Price comparison
        price_diff = centre_df['Sales Price(Kg)'].mean() - other_df['Sales Price(Kg)'].mean()
        price_ratio = centre_df['Sales Price(Kg)'].mean() / other_df['Sales Price(Kg)'].mean()
        
        insights.append(f"Regional Comparison ({region}):")
        insights.append(f"  • Average price is {abs(price_diff):.2f} ₹/Kg {'higher' if price_diff > 0 else 'lower'} than {other_type}")
        insights.append(f"  • Price ratio ({tea_type}/{other_type}): {price_ratio:.2f}")
        
        # Volume comparison
        centre_vol = centre_df['Sold Qty (Ton)'].sum()
        other_vol = other_df['Sold Qty (Ton)'].sum()
        vol_ratio = centre_vol / other_vol if other_vol > 0 else float('inf')
        
        insights.append(f"\nVolume Comparison:")
        insights.append(f"  • Total volume ratio ({tea_type}/{other_type}): {vol_ratio:.2f}")
        
        # Efficiency comparison
        centre_eff = centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum())
        other_eff = other_df['Sold Qty (Ton)'].sum() / (other_df['Sold Qty (Ton)'].sum() + other_df['Unsold Qty (Ton)'].sum())
        
        insights.append(f"\nEfficiency Comparison:")
        insights.append(f"  • Market efficiency: {centre_eff*100:.1f}% vs {other_eff*100:.1f}% for {other_type}")
    
    return insights

def generate_ai_narrative(df: pd.DataFrame, centre: str) -> str:
    """Generate AI-powered narrative market analysis"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return "AI narrative generation unavailable: OpenAI API key not found in environment variables."
        
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Prepare market data summary
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        latest_sale = centre_df['Sale No'].max()
        latest_data = centre_df[centre_df['Sale No'] == latest_sale].iloc[0]
        
        # Calculate key metrics
        price_change = centre_df['Sales Price(Kg)'].pct_change().mean()
        volume_change = (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']).pct_change().mean()
        efficiency = latest_data['Sold Qty (Ton)'] / (latest_data['Sold Qty (Ton)'] + latest_data['Unsold Qty (Ton)'])
        
        # Create market context
        market_context = f"""
        Market: {centre}
        Latest Sale: {latest_sale}
        Current Price: ₹{latest_data['Sales Price(Kg)']:.2f}/Kg
        Price Trend: {price_change*100:.1f}% average change
        Volume Trend: {volume_change*100:.1f}% average change
        Market Efficiency: {efficiency*100:.1f}%
        """
        
        # Generate AI analysis
        prompt = f"""You are a tea market analyst. Analyze the following tea market data and provide a concise, insightful narrative analysis focusing on key trends and market implications. Use a professional tone.

Market Data:
{market_context}

Provide a 2-3 sentence analysis highlighting the most important insights and potential market implications."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in AI narrative generation: {str(e)}")
        return f"AI narrative generation encountered an error: {str(e)}"

def generate_price_analysis(df: pd.DataFrame, centre: str) -> str:
    """Generate detailed price analysis including trends, seasonality, and forecasts"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return fallback_price_analysis(df, centre)
        
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate key price metrics
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        avg_price = centre_df['Sales Price(Kg)'].mean()
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        price_volatility = centre_df['Sales Price(Kg)'].std()
        
        market_context = f"""
        Market: {centre}
        Current Price: ₹{current_price:.2f}/Kg
        Average Price: ₹{avg_price:.2f}/Kg
        Price Trend: {price_trend*100:.1f}% average change
        Price Volatility: ₹{price_volatility:.2f}/Kg
        Historical Prices: {', '.join([f'₹{p:.2f}' for p in centre_df['Sales Price(Kg)'].tolist()])}
        """
        
        prompt = f"""You are a tea market price analyst. Analyze the following price data and provide detailed insights on:
1. Price trends and their implications
2. Price seasonality patterns if any
3. Short-term price forecast based on current trends
4. Price volatility analysis

Market Data:
{market_context}

Provide a comprehensive analysis in 3-4 sentences."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in price analysis: {str(e)}")
        return fallback_price_analysis(df, centre)

def generate_market_insights(df: pd.DataFrame, centre: str) -> str:
    """Generate market position and competitive analysis insights"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return fallback_market_insights(df, centre)
        
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        region, tea_type = centre.split(' CTC ')
        
        # Calculate market metrics
        market_share = centre_df['Sold Qty (Ton)'].sum() / df['Sold Qty (Ton)'].sum()
        efficiency = centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum())
        
        market_context = f"""
        Market: {centre}
        Region: {region}
        Tea Type: {tea_type}
        Market Share: {market_share*100:.1f}%
        Market Efficiency: {efficiency*100:.1f}%
        """
        
        prompt = f"""You are a tea market analyst. Analyze the following market data and provide insights on:
1. Market positioning
2. Competitive advantages/disadvantages
3. Market efficiency analysis
4. Strategic market position

Market Data:
{market_context}

Provide a comprehensive analysis in 3-4 sentences."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in market insights: {str(e)}")
        return fallback_market_insights(df, centre)

def generate_volume_analysis(df: pd.DataFrame, centre: str) -> str:
    """Generate detailed volume analysis including trends and demand patterns"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return fallback_volume_analysis(df, centre)
        
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate volume metrics
        total_volume = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
        avg_volume = total_volume.mean()
        volume_trend = total_volume.pct_change().mean()
        sold_ratio = centre_df['Sold Qty (Ton)'].sum() / total_volume.sum()
        
        volume_context = f"""
        Market: {centre}
        Average Volume: {avg_volume:.1f} tons
        Volume Trend: {volume_trend*100:.1f}% average change
        Sold Ratio: {sold_ratio*100:.1f}%
        Recent Volumes: {', '.join([f'{v:.0f}' for v in total_volume.tail(5).tolist()])} tons
        """
        
        prompt = f"""You are a tea market volume analyst. Analyze the following volume data and provide insights on:
1. Volume trends and patterns
2. Demand-supply dynamics
3. Volume seasonality if any
4. Market absorption capacity

Market Data:
{volume_context}

Provide a comprehensive analysis in 3-4 sentences."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in volume analysis: {str(e)}")
        return fallback_volume_analysis(df, centre)

def generate_recommendations(df: pd.DataFrame, centre: str) -> str:
    """Generate AI-powered strategic recommendations"""
    try:
        if 'OPENAI_API_KEY' not in os.environ:
            return fallback_recommendations(df, centre)
        
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate key metrics for recommendations
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        volume_trend = (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']).pct_change().mean()
        efficiency = centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)']).sum()
        
        context = f"""
        Market: {centre}
        Price Trend: {price_trend*100:.1f}% average change
        Volume Trend: {volume_trend*100:.1f}% average change
        Market Efficiency: {efficiency*100:.1f}%
        """
        
        prompt = f"""You are a tea market strategist. Based on the following market data, provide strategic recommendations focusing on:
1. Price positioning strategy
2. Volume optimization
3. Market efficiency improvement
4. Competitive positioning

Market Data:
{context}

Provide 3-4 specific, actionable recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in recommendations: {str(e)}")
        return fallback_recommendations(df, centre)

# Fallback analysis functions for when AI is unavailable
def fallback_price_analysis(df: pd.DataFrame, centre: str) -> str:
    centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
    current_price = centre_df['Sales Price(Kg)'].iloc[-1]
    avg_price = centre_df['Sales Price(Kg)'].mean()
    price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
    return f"""Price Analysis (Data-based):
• Current price: ₹{current_price:.2f}/Kg
• Average price: ₹{avg_price:.2f}/Kg
• Price trend: {price_trend*100:.1f}% average change
• Trend indication: {'Upward' if price_trend > 0 else 'Downward'} movement"""

def fallback_market_insights(df: pd.DataFrame, centre: str) -> str:
    centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
    market_share = centre_df['Sold Qty (Ton)'].sum() / df['Sold Qty (Ton)'].sum()
    efficiency = centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum())
    return f"""Market Insights (Data-based):
• Market share: {market_share*100:.1f}%
• Market efficiency: {efficiency*100:.1f}%
• Position: {'Strong' if efficiency > 0.8 else 'Moderate' if efficiency > 0.6 else 'Weak'} market performance"""

def fallback_volume_analysis(df: pd.DataFrame, centre: str) -> str:
    centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
    total_volume = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
    avg_volume = total_volume.mean()
    volume_trend = total_volume.pct_change().mean()
    return f"""Volume Analysis (Data-based):
• Average volume: {avg_volume:.1f} tons
• Volume trend: {volume_trend*100:.1f}% average change
• Trend indication: {'Increasing' if volume_trend > 0 else 'Decreasing'} volume"""

def fallback_recommendations(df: pd.DataFrame, centre: str) -> str:
    centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
    efficiency = centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum())
    price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
    return f"""Recommendations (Data-based):
• {'Maintain' if efficiency > 0.8 else 'Improve'} current market efficiency
• {'Maintain' if price_trend > 0 else 'Review'} pricing strategy
• Focus on {'volume optimization' if efficiency < 0.7 else 'price optimization'}"""

def generate_insights(df: pd.DataFrame) -> List[str]:
    """Generate comprehensive market insights with specialized analysis for each category"""
    all_insights = []
    
    try:
        for centre in sorted(df['Centre'].unique()):
            region, tea_type = centre.split(' CTC ')
            all_insights.append(f"\n=== {region} CTC {tea_type} Analysis ===\n")
            
            # Price Analysis
            price_analysis = generate_price_analysis(df, centre)
            all_insights.append("--- Price Analysis ---")
            all_insights.append(price_analysis)
            
            # Market Insights
            market_analysis = generate_market_insights(df, centre)
            all_insights.append("\n--- Market Insights ---")
            all_insights.append(market_analysis)
            
            # Volume Analysis
            volume_analysis = generate_volume_analysis(df, centre)
            all_insights.append("\n--- Volume Analysis ---")
            all_insights.append(volume_analysis)
            
            # Recommendations
            recommendations = generate_recommendations(df, centre)
            all_insights.append("\n--- Strategic Recommendations ---")
            all_insights.append(recommendations)
            
            all_insights.append("\n" + "="*50 + "\n")
        
        return all_insights
    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}")
        return [f"Error generating insights: {str(e)}"]