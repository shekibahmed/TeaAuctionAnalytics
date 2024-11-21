import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple, Dict
import openai
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_closest_match(column: str,
                      possible_matches: List[str]) -> Tuple[str, float]:
    """Find the closest matching column name based on string similarity."""
    from difflib import SequenceMatcher

    column_lower = column.lower().strip()
    scores = [(match, SequenceMatcher(None, column_lower,
                                      match.lower().strip()).ratio())
              for match in possible_matches]
    return max(scores, key=lambda x: x[1])


def standardize_market_category(category: str) -> str:
    """Standardize market category name to the format: {region} CTC {type}."""
    category = ' '.join(category.strip().split())

    if ' CTC ' in category:
        region, type_part = category.split(' CTC ')
    else:
        parts = category.split()
        if len(parts) < 3 or parts[0] not in [
                'North', 'South'
        ] or parts[1] != 'India' or parts[-1] not in ['Leaf', 'Dust']:
            return category
        region = ' '.join(parts[:2])
        type_part = parts[-1]

    if region not in ['North India', 'South India']:
        return category
    if type_part not in ['Leaf', 'Dust']:
        return category

    return f"{region} CTC {type_part}"


def process_excel_data(file) -> pd.DataFrame:
    """Process uploaded Excel file and return formatted DataFrame."""
    column_variations = {
        'Centre':
        ['Centre', 'Center', 'Market Center', 'Market Centre', 'Location'],
        'Sale No': ['Sale No', 'Sale Number', 'Sale_No', 'SaleNo', 'Sale'],
        'Sales Price(Kg)':
        ['Sales Price(Kg)', 'Price/Kg', 'Sales Price', 'Price'],
        'Sold Qty (Ton)':
        ['Sold Qty (Ton)', 'Sold Quantity', 'Sold_Qty', 'Quantity Sold'],
        'Unsold Qty (Ton)': [
            'Unsold Qty (Ton)', 'Unsold Quantity', 'Unsold_Qty',
            'Quantity Unsold'
        ]
    }

    required_columns = list(column_variations.keys())

    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

    column_mapping = {}
    missing_columns = []

    for required_col in required_columns:
        variations = column_variations[required_col]
        found = False

        for var in variations:
            if var in df.columns:
                column_mapping[var] = required_col
                found = True
                break

        if not found:
            potential_matches = [(col, get_closest_match(col, variations))
                                 for col in df.columns
                                 if col not in column_mapping.keys()]
            best_match = max(
                potential_matches,
                key=lambda x: x[1][1]) if potential_matches else (None, (None,
                                                                         0))

            if best_match[1][1] > 0.8:
                column_mapping[best_match[0]] = required_col
            else:
                missing_columns.append(required_col)

    if missing_columns:
        error_msg = "\nMissing required columns:\n"
        for col in missing_columns:
            error_msg += f"- {col} (accepted variations: {', '.join(column_variations[col])})\n"
        raise ValueError(error_msg)

    df = df.rename(columns=column_mapping)
    df = df[required_columns].dropna()

    df['Sale No'] = pd.to_numeric(df['Sale No'])
    df['Sales Price(Kg)'] = pd.to_numeric(df['Sales Price(Kg)'])
    df['Sold Qty (Ton)'] = pd.to_numeric(df['Sold Qty (Ton)'])
    df['Unsold Qty (Ton)'] = pd.to_numeric(df['Unsold Qty (Ton)'])
    df['Centre'] = df['Centre'].apply(standardize_market_category)

    return df


def generate_market_insights(df: pd.DataFrame, centre: str) -> str:
    """
    Generate market insights for a specific centre.
    
    Args:
        df (pd.DataFrame): Input dataframe with market data
        centre (str): Centre name to analyze
    
    Returns:
        str: Market insights analysis text
    """
    try:
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate key metrics
        total_volume = centre_df['Sold Qty (Ton)'].sum()
        avg_price = centre_df['Sales Price(Kg)'].mean()
        price_volatility = centre_df['Sales Price(Kg)'].std()
        market_efficiency = (centre_df['Sold Qty (Ton)'] / 
                           (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])).mean()
        
        # Generate insights
        insights = [
            f"Market Overview for {centre}:",
            f"• Total Trading Volume: {total_volume:.2f} tons",
            f"• Average Price: ₹{avg_price:.2f}/kg",
            f"• Price Volatility: ₹{price_volatility:.2f}/kg",
            f"• Market Efficiency: {market_efficiency*100:.1f}%"
        ]
        
        # Recent trends
        recent_df = centre_df.tail(5)
        price_trend = "increasing" if recent_df['Sales Price(Kg)'].is_monotonic_increasing else \
                     "decreasing" if recent_df['Sales Price(Kg)'].is_monotonic_decreasing else \
                     "fluctuating"
        
        insights.append(f"\nRecent Market Trends:")
        insights.append(f"• Prices have been {price_trend} in recent sales")
        
        return "\n".join(insights)
        
    except Exception as e:
        logging.error(f"Error generating market insights: {str(e)}")
        return "Error generating market insights"

def generate_price_analysis(df: pd.DataFrame, centre: str) -> str:
    """
    Generate price analysis for a specific centre.
    
    Args:
        df (pd.DataFrame): Input dataframe with market data
        centre (str): Centre name to analyze
    
    Returns:
        str: Price analysis text
    """
    try:
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate price metrics
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        avg_price = centre_df['Sales Price(Kg)'].mean()
        price_range = centre_df['Sales Price(Kg)'].max() - centre_df['Sales Price(Kg)'].min()
        price_volatility = centre_df['Sales Price(Kg)'].std()
        
        # Calculate price trends
        recent_df = centre_df.tail(5)
        price_change = ((recent_df['Sales Price(Kg)'].iloc[-1] / 
                        recent_df['Sales Price(Kg)'].iloc[0] - 1) * 100)
        
        analysis = [
            f"Price Analysis for {centre}:",
            f"• Current Price: ₹{current_price:.2f}/kg",
            f"• Average Price: ₹{avg_price:.2f}/kg",
            f"• Price Range: ₹{price_range:.2f}/kg",
            f"• Price Volatility: ₹{price_volatility:.2f}/kg",
            f"\nRecent Price Trends:",
            f"• 5-Sale Price Change: {price_change:+.1f}%"
        ]
        
        return "\n".join(analysis)
        
    except Exception as e:
        logging.error(f"Error generating price analysis: {str(e)}")
        return "Error generating price analysis"

def analyze_comparatives(df: pd.DataFrame, centre: str) -> list:
    """Analyze market comparatives with enhanced error handling and large dataset support."""
    try:
        # Extract region and tea type from centre name
        region, tea_type = centre.split(' CTC ')
        other_type = 'Leaf' if tea_type == 'Dust' else 'Dust'
        
        # Filter data for both categories with error handling
        current_data = df[df['Centre'].str.contains(tea_type, na=False) & 
                         df['Centre'].str.contains(region, na=False)].copy()
        other_data = df[df['Centre'].str.contains(other_type, na=False) & 
                       df['Centre'].str.contains(region, na=False)].copy()
        
        logging.debug(f"Processing {len(current_data)} {tea_type} records")
        logging.debug(f"Processing {len(other_data)} {other_type} records")
        
        if current_data.empty:
            return [f"No data available for {tea_type} category in {region}"]
        if other_data.empty:
            return [f"No data available for {other_type} category in {region}"]
        
        # Calculate weighted average prices using batch processing
        def calculate_weighted_price(data: pd.DataFrame) -> float:
            if len(data) == 0:
                return 0
                
            total_weighted_sum = 0
            total_weight = 0
            batch_size = 1000
            
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                weights = batch['Sold Qty (Ton)'].to_numpy()
                prices = batch['Sales Price(Kg)'].to_numpy()
                
                # Filter out NaN values
                mask = np.isfinite(weights) & np.isfinite(prices)
                valid_weights = weights[mask]
                valid_prices = prices[mask]
                
                total_weighted_sum += np.sum(valid_weights * valid_prices)
                total_weight += np.sum(valid_weights)
            
            return total_weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate metrics
        current_price = calculate_weighted_price(current_data)
        other_price = calculate_weighted_price(other_data)
        price_diff = current_price - other_price
        
        return [
            f"Average {tea_type} Price: ₹{current_price:.2f}/Kg",
            f"Average {other_type} Price: ₹{other_price:.2f}/Kg",
            f"Price difference: ₹{abs(price_diff):.2f}/Kg ({'higher' if price_diff > 0 else 'lower'})"
        ]
        
    except Exception as e:
        logging.error(f"Error in comparative analysis: {str(e)}")
        return [f"Error in comparative analysis: {str(e)}"]

def generate_price_analysis(df: pd.DataFrame, centre: str) -> str:
    """Generate detailed price analysis including trends and comparisons."""
    try:
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()

        # Calculate current and weighted prices
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        weighted_avg_price = ((centre_df['Sales Price(Kg)'] *
                               centre_df['Sold Qty (Ton)']).sum() /
                              centre_df['Sold Qty (Ton)'].sum())
        price_range = centre_df['Sales Price(Kg)'].max(
        ) - centre_df['Sales Price(Kg)'].min()
        price_percentile = (centre_df['Sales Price(Kg)']
                            <= current_price).mean() * 100

        insights = [
            f"Current Price: ₹{current_price:.2f}/Kg",
            f"Weighted Average Price: ₹{weighted_avg_price:.2f}/Kg",
            f"Price Range: ₹{price_range:.2f}/Kg",
            f"Current Price Percentile: {price_percentile:.1f}%",
        ]

        return "\n".join(insights)

    except Exception as e:
        return f"Error in generating price analysis: {str(e)}"


def analyze_comparatives(df: pd.DataFrame, centre: str) -> List[str]:
    """Analyze cross-market comparisons and benchmarking with efficient batch processing."""
    insights = []
    BATCH_SIZE = 1000  # Process data in chunks to handle large datasets

    if ' CTC ' not in centre:
        return [f"Invalid centre format: {centre}. Expected format: 'Region CTC Type'"]

    region, tea_type = centre.split(' CTC ')
    other_type = 'Leaf' if tea_type == 'Dust' else 'Dust'

    try:
        # Use exact matching with standardized centre names for better accuracy
        dust_data = df[df['Centre'] == f"{region} CTC Dust"].copy()
        leaf_data = df[df['Centre'] == f"{region} CTC Leaf"].copy()

        # Validate data availability
        if dust_data.empty and tea_type == 'Dust':
            return [f"No data available for Dust category in {region}"]
        if leaf_data.empty and tea_type == 'Leaf':
            return [f"No data available for Leaf category in {region}"]

        def calculate_weighted_price(data: pd.DataFrame) -> float:
            """Calculate weighted average price using batch processing."""
            if len(data) == 0:
                return 0.0
                
            total_weighted_sum = 0.0
            total_weight = 0.0
            
            # Process data in batches
            for start_idx in range(0, len(data), BATCH_SIZE):
                batch = data.iloc[start_idx:start_idx + BATCH_SIZE]
                weights = batch['Sold Qty (Ton)'].to_numpy()
                prices = batch['Sales Price(Kg)'].to_numpy()
                
                # Handle NaN values
                valid_mask = np.isfinite(weights) & np.isfinite(prices)
                valid_weights = weights[valid_mask]
                valid_prices = prices[valid_mask]
                
                batch_sum = np.sum(valid_weights * valid_prices)
                batch_weight = np.sum(valid_weights)
                
                total_weighted_sum += batch_sum
                total_weight += batch_weight
            
            return total_weighted_sum / total_weight if total_weight > 0 else 0.0

        def calculate_market_metrics(data: pd.DataFrame) -> Dict[str, float]:
            """Calculate market metrics using batch processing."""
            metrics = {'volume': 0.0, 'efficiency': 0.0}
            
            total_sold = 0.0
            total_unsold = 0.0
            
            for start_idx in range(0, len(data), BATCH_SIZE):
                batch = data.iloc[start_idx:start_idx + BATCH_SIZE]
                sold = batch['Sold Qty (Ton)'].sum()
                unsold = batch['Unsold Qty (Ton)'].sum()
                
                total_sold += sold
                total_unsold += unsold
            
            metrics['volume'] = total_sold
            total = total_sold + total_unsold
            metrics['efficiency'] = (total_sold / total) if total > 0 else 0.0
            
            return metrics

        # Calculate metrics for both categories
        dust_price = calculate_weighted_price(dust_data)
        leaf_price = calculate_weighted_price(leaf_data)
        dust_metrics = calculate_market_metrics(dust_data)
        leaf_metrics = calculate_market_metrics(leaf_data)

        # Generate insights
        insights.extend([
            f"Price Analysis ({region}):",
            f"  • Dust weighted average price: ₹{dust_price:.2f}/Kg",
            f"  • Leaf weighted average price: ₹{leaf_price:.2f}/Kg",
            f"  • Price difference (Dust - Leaf): ₹{(dust_price - leaf_price):.2f}/Kg",
            f"  • Price ratio (Dust/Leaf): {(dust_price / leaf_price if leaf_price > 0 else 0):.2f}",
            f"\nVolume Analysis:",
            f"  • Dust total volume: {dust_metrics['volume']:,.0f} tons",
            f"  • Leaf total volume: {leaf_metrics['volume']:,.0f} tons",
            f"  • Volume ratio (Dust/Leaf): {(dust_metrics['volume'] / leaf_metrics['volume'] if leaf_metrics['volume'] > 0 else 0):.2f}",
            f"\nMarket Efficiency:",
            f"  • Dust: {dust_metrics['efficiency']*100:.1f}%",
            f"  • Leaf: {leaf_metrics['efficiency']*100:.1f}%"
        ])

    except Exception as e:
        logging.error(f"Error in comparative analysis: {str(e)}")
        insights.append(f"Error during comparative analysis: {str(e)}")

    return insights

def generate_ai_narrative(df: pd.DataFrame, centre: str) -> str:
    '''
    Generate AI-powered narrative analysis of market data.
    
    Args:
        df (pd.DataFrame): Input dataframe with market data
        centre (str): Centre name to analyze
    
    Returns:
        str: Generated narrative analysis
    '''
    try:
        centre_df = df[df['Centre'] == centre].sort_values('Sale No').copy()
        
        # Calculate key metrics for narrative
        current_price = centre_df['Sales Price(Kg)'].iloc[-1]
        avg_price = centre_df['Sales Price(Kg)'].mean()
        price_trend = "increasing" if centre_df['Sales Price(Kg)'].iloc[-3:].is_monotonic_increasing else \
                     "decreasing" if centre_df['Sales Price(Kg)'].iloc[-3:].is_monotonic_decreasing else \
                     "fluctuating"
        
        volume_change = ((centre_df['Sold Qty (Ton)'].iloc[-1] / 
                         centre_df['Sold Qty (Ton)'].iloc[-2] - 1) * 100)
        
        narrative = [
            f"Market Analysis for {centre}:",
            f"The current market price is ₹{current_price:.2f}/kg, compared to an average of ₹{avg_price:.2f}/kg.",
            f"Prices have been {price_trend} in recent sales.",
            f"Trading volume has {'increased' if volume_change > 0 else 'decreased'} by {abs(volume_change):.1f}% in the latest sale."
        ]
        
        return " ".join(narrative)
        
    except Exception as e:
        logging.error(f"Error generating AI narrative: {str(e)}")
        return "Error generating market narrative"

def analyze_levels(df: pd.DataFrame, centre: str) -> dict:
    '''
    Analyze price and volume levels for a given centre.
    
    Args:
        df (pd.DataFrame): Input dataframe with market data
        centre (str): Centre name to analyze
    
    Returns:
        dict: Dictionary containing level analysis results
    '''
    try:
        centre_df = df[df['Centre'] == centre].copy()
        if centre_df.empty:
            return {
                'error': f'No data available for centre: {centre}'
            }

        # Calculate price levels
        price_stats = {
            'mean': centre_df['Sales Price(Kg)'].mean(),
            'median': centre_df['Sales Price(Kg)'].median(),
            'std': centre_df['Sales Price(Kg)'].std(),
            'min': centre_df['Sales Price(Kg)'].min(),
            'max': centre_df['Sales Price(Kg)'].max()
        }

        # Define price bands
        price_bands = {
            'low': price_stats['mean'] - price_stats['std'],
            'medium': price_stats['mean'],
            'high': price_stats['mean'] + price_stats['std']
        }

        # Calculate volume thresholds using percentiles
        volume_thresholds = {
            'low': centre_df['Sold Qty (Ton)'].quantile(0.25),
            'medium': centre_df['Sold Qty (Ton)'].quantile(0.50),
            'high': centre_df['Sold Qty (Ton)'].quantile(0.75)
        }

        # Calculate market efficiency
        total_volume = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
        market_efficiency = (centre_df['Sold Qty (Ton)'].sum() / total_volume.sum()) * 100

        # Calculate price distribution
        price_distribution = {
            'below_mean': (centre_df['Sales Price(Kg)'] < price_stats['mean']).mean() * 100,
            'above_mean': (centre_df['Sales Price(Kg)'] > price_stats['mean']).mean() * 100
        }

        # Prepare the final analysis results
        analysis_results = {
            'price_statistics': price_stats,
            'price_bands': price_bands,
            'volume_thresholds': volume_thresholds,
            'market_efficiency': market_efficiency,
            'price_distribution': price_distribution,
            'data_points': len(centre_df),
            'analysis_period': {
                'start': centre_df['Sale No'].min(),
                'end': centre_df['Sale No'].max()
            }
        }

        return analysis_results

    except Exception as e:
        logging.error(f"Error in level analysis: {str(e)}")
        return {'error': f'Analysis failed: {str(e)}'}