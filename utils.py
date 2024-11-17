import pandas as pd
import numpy as np
from typing import List, Dict

def process_excel_data(file):
    """Process uploaded Excel file and return formatted DataFrame"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    required_columns = ['Centre', 'Sale No', 'Sales Price(Kg)', 'Sold Qty (Ton)', 'Unsold Qty (Ton)']
    
    # Validate columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Upload file must contain all required columns: Centre, Sale No, Sales Price(Kg), Sold Qty (Ton), Unsold Qty (Ton)")
    
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

def generate_insights(df: pd.DataFrame) -> List[str]:
    """Generate comprehensive market insights with levels, trends, and comparatives"""
    all_insights = []
    
    for centre in sorted(df['Centre'].unique()):
        region, tea_type = centre.split(' CTC ')
        all_insights.append(f"\n=== {region} CTC {tea_type} Analysis ===\n")
        
        # Level Analysis
        all_insights.append("--- Market Levels ---")
        all_insights.extend(analyze_levels(df, centre))
        
        # Trend Analysis
        all_insights.append("\n--- Market Trends ---")
        all_insights.extend(analyze_trends(df, centre))
        
        # Comparative Analysis
        all_insights.append("\n--- Market Comparatives ---")
        all_insights.extend(analyze_comparatives(df, centre))
        
        all_insights.append("\n" + "="*50 + "\n")
    
    return all_insights
