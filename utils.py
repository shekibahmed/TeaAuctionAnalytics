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
    
    return df

def generate_insights(df):
    """Generate automated insights from the data per centre"""
    insights = []
    
    # Generate insights for each centre
    for centre in df['Centre'].unique():
        centre_df = df[df['Centre'] == centre]
        
        insights.append(f"Analysis for {centre}:")
        
        # Price trends
        price_trend = centre_df['Sales Price(Kg)'].pct_change().mean()
        if price_trend > 0:
            insights.append(f"  • Positive price trend with {price_trend*100:.1f}% average growth")
        else:
            insights.append(f"  • Negative price trend with {abs(price_trend*100):.1f}% average decline")
        
        # Sales volume analysis
        avg_sold = centre_df['Sold Qty (Ton)'].mean()
        latest_sold = centre_df['Sold Qty (Ton)'].iloc[-1]
        if latest_sold > avg_sold:
            insights.append(f"  • Latest sales volume ({latest_sold:,.0f} tons) is above the average ({avg_sold:,.0f} tons)")
        else:
            insights.append(f"  • Latest sales volume ({latest_sold:,.0f} tons) is below the average ({avg_sold:,.0f} tons)")
        
        # Market efficiency
        total_qty = centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum()
        sold_ratio = centre_df['Sold Qty (Ton)'].sum() / total_qty if total_qty > 0 else 0
        insights.append(f"  • Market efficiency: {sold_ratio*100:.1f}% of total quantity sold")
        
        insights.append("")  # Add spacing between centres
    
    return insights
