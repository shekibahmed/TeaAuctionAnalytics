import pandas as pd
import numpy as np

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

def generate_insights(df):
    """Generate automated insights from the data per centre"""
    insights = []
    
    # Generate insights for each centre
    for centre in sorted(df['Centre'].unique()):
        centre_df = df[df['Centre'] == centre].copy()
        
        # Extract region and type from centre name
        region = centre.split(' CTC ')[0]
        tea_type = centre.split(' CTC ')[1]
        
        insights.append(f"Analysis for {region} CTC {tea_type}:")
        
        # Price trends
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        if pd.notna(price_trend):
            if price_trend > 0:
                insights.append(f"  • Positive price trend with {price_trend*100:.1f}% average growth")
            else:
                insights.append(f"  • Negative price trend with {abs(price_trend*100):.1f}% average decline")
        
        # Sales volume analysis
        avg_sold = centre_df['Sold Qty (Ton)'].mean()
        latest_sold = centre_df['Sold Qty (Ton)'].iloc[-1]
        if pd.notna(avg_sold) and pd.notna(latest_sold):
            if latest_sold > avg_sold:
                insights.append(f"  • Latest sales volume ({latest_sold:,.0f} tons) is above the average ({avg_sold:,.0f} tons)")
            else:
                insights.append(f"  • Latest sales volume ({latest_sold:,.0f} tons) is below the average ({avg_sold:,.0f} tons)")
        
        # Market efficiency
        total_qty = centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum()
        sold_ratio = centre_df['Sold Qty (Ton)'].sum() / total_qty if total_qty > 0 else 0
        if pd.notna(sold_ratio):
            insights.append(f"  • Market efficiency: {sold_ratio*100:.1f}% of total quantity sold")
        
        # Compare with other type in same region
        other_type = 'Dust' if tea_type == 'Leaf' else 'Leaf'
        other_centre = f"{region} CTC {other_type}"
        if other_centre in df['Centre'].unique():
            other_df = df[df['Centre'] == other_centre]
            price_diff = (centre_df['Sales Price(Kg)'].mean() - other_df['Sales Price(Kg)'].mean())
            if pd.notna(price_diff):
                insights.append(f"  • Price differential with {other_type}: {abs(price_diff):.2f} ₹/Kg {'higher' if price_diff > 0 else 'lower'}")
        
        insights.append("")  # Add spacing between centres
    
    return insights
