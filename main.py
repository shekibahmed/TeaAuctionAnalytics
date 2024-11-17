import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import process_excel_data, generate_insights
from styles import apply_custom_styles

# Setup page config
st.set_page_config(page_title="CTC Tea Sales Analytics Dashboard", layout="wide")
apply_custom_styles()

st.title("CTC Tea Sales Analytics Dashboard")

# File upload section
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls', 'csv'])

# Load default or uploaded data
try:
    if uploaded_file is not None:
        try:
            df = process_excel_data(uploaded_file)
            st.success("Successfully loaded uploaded file")
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.info("Loading default data instead")
            df = pd.read_csv('assets/default_data.csv')
    else:
        st.info("Using default data")
        df = pd.read_csv('assets/default_data.csv')
    
    # Ensure proper data types
    df['Sale No'] = pd.to_numeric(df['Sale No'])
    df['Sales Price(Kg)'] = pd.to_numeric(df['Sales Price(Kg)'])
    df['Sold Qty (Ton)'] = pd.to_numeric(df['Sold Qty (Ton)'])
    df['Unsold Qty (Ton)'] = pd.to_numeric(df['Unsold Qty (Ton)'])
    
    # Center selection with region and type filtering
    regions = sorted(list(set([centre.split(' CTC ')[0] for centre in df['Centre'].unique()])))
    tea_types = sorted(list(set([centre.split(' CTC ')[1] for centre in df['Centre'].unique()])))
    
    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect(
            "Select Regions",
            options=regions,
            default=regions,
            key='region_selector'
        )
    
    with col2:
        selected_types = st.multiselect(
            "Select Tea Types",
            options=tea_types,
            default=tea_types,
            key='type_selector'
        )
    
    # Filter centres based on region and type selection
    selected_centres = sorted([
        centre for centre in df['Centre'].unique()
        if any(region in centre for region in selected_regions)
        and any(tea_type in centre for tea_type in selected_types)
    ])
    
    if not selected_centres:
        st.warning("Please select at least one region and tea type.")
        st.stop()
    
    # Filter data for selected centres
    df_selected = df[df['Centre'].isin(selected_centres)].copy()
    
    # Create charts based on selection
    num_centres = len(selected_centres)
    if num_centres == 1:
        # Single center - larger chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        centre = selected_centres[0]
        centre_df = df_selected[df_selected['Centre'] == centre].copy()
        
        # Extract region and type for title
        region, tea_type = centre.split(' CTC ')
        
        # Sort by Sale No
        centre_df = centre_df.sort_values('Sale No')
        
        # Bar charts for Sold and Unsold
        fig.add_trace(
            go.Bar(
                name='Sold Qty (Ton)',
                x=centre_df['Sale No'],
                y=centre_df['Sold Qty (Ton)'],
                marker_color='#FF9966'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                name='Unsold Qty (Ton)',
                x=centre_df['Sale No'],
                y=centre_df['Unsold Qty (Ton)'],
                marker_color='#808080'
            ),
            secondary_y=False
        )
        
        # Line chart for Sales Price
        fig.add_trace(
            go.Scatter(
                name='Sales Price (Kg)',
                x=centre_df['Sale No'],
                y=centre_df['Sales Price(Kg)'],
                line=dict(color='#3366CC', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout for single chart
        fig.update_layout(
            title=f"{region} CTC {tea_type} Market Trends",
            height=600,
            barmode='group',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text='Sale No')
        fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False)
        fig.update_yaxes(title_text='Price (₹/Kg)', secondary_y=True)
    
    else:
        # Multiple centers - grid layout
        rows = (num_centres + 1) // 2
        fig = make_subplots(
            rows=rows,
            cols=min(2, num_centres),
            specs=[[{"secondary_y": True}] * min(2, num_centres)] * rows,
            subplot_titles=[f"{centre.split(' CTC ')[0]} CTC {centre.split(' CTC ')[1]} Market Trends" 
                          for centre in selected_centres]
        )
        
        for idx, centre in enumerate(selected_centres):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            centre_df = df_selected[df_selected['Centre'] == centre].copy()
            centre_df = centre_df.sort_values('Sale No')
            
            # Bar charts for Sold and Unsold
            fig.add_trace(
                go.Bar(
                    name='Sold Qty (Ton)',
                    x=centre_df['Sale No'],
                    y=centre_df['Sold Qty (Ton)'],
                    marker_color='#FF9966',
                    showlegend=(idx == 0)
                ),
                row=row, col=col,
                secondary_y=False
            )
            
            fig.add_trace(
                go.Bar(
                    name='Unsold Qty (Ton)',
                    x=centre_df['Sale No'],
                    y=centre_df['Unsold Qty (Ton)'],
                    marker_color='#808080',
                    showlegend=(idx == 0)
                ),
                row=row, col=col,
                secondary_y=False
            )
            
            # Line chart for Sales Price
            fig.add_trace(
                go.Scatter(
                    name='Sales Price (Kg)',
                    x=centre_df['Sale No'],
                    y=centre_df['Sales Price(Kg)'],
                    line=dict(color='#3366CC', width=2),
                    showlegend=(idx == 0)
                ),
                row=row, col=col,
                secondary_y=True
            )
        
        # Update layout for multiple charts
        fig.update_layout(
            height=400 * rows,
            barmode='group',
            hovermode='x unified',
            template='plotly_white',
            margin=dict(t=50, b=20, l=60, r=60)
        )
        
        # Update axes labels for all subplots
        for i in range(1, rows + 1):
            for j in range(1, min(3, num_centres + 1)):
                if (i-1)*2 + j <= num_centres:
                    fig.update_xaxes(title_text='Sale No', row=i, col=j)
                    fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False, row=i, col=j)
                    fig.update_yaxes(title_text='Price (₹/Kg)', secondary_y=True, row=i, col=j)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Metrics - only for selected centres
    st.subheader("Key Metrics by Market")
    metrics_cols = st.columns(len(selected_centres))
    
    for idx, (col, centre) in enumerate(zip(metrics_cols, selected_centres)):
        centre_df = df_selected[df_selected['Centre'] == centre].copy()
        region, tea_type = centre.split(' CTC ')
        
        with col:
            st.markdown(f"**{region} CTC {tea_type}**")
            
            # Calculate metrics safely handling NaN values
            avg_price = centre_df['Sales Price(Kg)'].mean()
            price_change = centre_df['Sales Price(Kg)'].pct_change().mean()
            total_sold = centre_df['Sold Qty (Ton)'].sum()
            sold_change = centre_df['Sold Qty (Ton)'].pct_change().mean()
            total_qty = total_sold + centre_df['Unsold Qty (Ton)'].sum()
            efficiency = (total_sold / total_qty * 100) if total_qty > 0 else 0
            
            st.metric(
                "Avg Sales Price",
                f"₹{avg_price:.2f}/Kg",
                f"{price_change*100:.1f}%" if pd.notna(price_change) else None
            )
            st.metric(
                "Total Sold Qty",
                f"{total_sold:,.0f} Tons",
                f"{sold_change*100:.1f}%" if pd.notna(sold_change) else None
            )
            st.metric(
                "Market Efficiency",
                f"{efficiency:.1f}%"
            )
    
    # Automated Insights - only for selected centres
    st.subheader("Market Insights")
    insights = generate_insights(df_selected)
    for insight in insights:
        st.info(insight)
    
    # Data Table - only for selected centres
    st.subheader("Detailed Data View")
    st.dataframe(
        df_selected.sort_values(['Centre', 'Sale No']),
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Error details:", e.__class__.__name__)
    import traceback
    st.code(traceback.format_exc())
