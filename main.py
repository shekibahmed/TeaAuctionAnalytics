import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import (process_excel_data, generate_price_analysis,
                   generate_market_insights, generate_ai_narrative,
                   analyze_levels, analyze_trends, analyze_comparatives,
                   calculate_correlations, analyze_key_correlations)
from styles import apply_custom_styles
import os

# Setup page config
st.set_page_config(page_title="CTC Tea Sales Analytics Dashboard",
                   layout="wide")
apply_custom_styles()

# Add title with description
st.title("Auction Analytics")
st.markdown("""
This dashboard provides comprehensive analysis of CTC tea sales across North and South India markets,
featuring AI-powered market insights and traditional market metrics.
""")

uploaded_file = st.file_uploader("Upload Excel File",
                                 type=['xlsx', 'xls', 'csv'])

try:
    if uploaded_file is not None:
        # Process uploaded file
        df = process_excel_data(uploaded_file)
        st.success("File uploaded and processed successfully!")

        # Center selection with region and type filtering
        regions = sorted(
            list(
                set([
                    centre.split(' CTC ')[0]
                    for centre in df['Centre'].unique()
                ])))
        tea_types = sorted(
            list(
                set([
                    centre.split(' CTC ')[1]
                    for centre in df['Centre'].unique()
                ])))

        col1, col2 = st.columns(2)
        with col1:
            selected_regions = st.multiselect("Select Regions",
                                              options=regions,
                                              default=["North India"],
                                              key='region_selector')

        with col2:
            selected_types = st.multiselect("Select Tea Types",
                                            options=tea_types,
                                            default=["Dust"],
                                            key='type_selector')

        # Filter centres based on region and type selection
        selected_centres = sorted([
            centre for centre in df['Centre'].unique()
            if any(region in centre for region in selected_regions) and any(
                tea_type in centre for tea_type in selected_types)
        ])

        if not selected_centres:
            st.warning("Please select at least one region and tea type.")
            st.stop()

        # Filter data for selected centres
        df_selected = df[df['Centre'].isin(selected_centres)].copy()

        # Create charts based on selection
        num_centres = len(selected_centres)
        if num_centres == 1:
            # Single center - larger chart with table
            fig = make_subplots(
                rows=2,
                cols=1,
                specs=[[{
                    "secondary_y": True
                }], [{
                    "type": "table"
                }]],
                row_heights=[0.7, 0.3],  # Adjusted ratio
                vertical_spacing=0.1  # Increased spacing
            )

            centre = selected_centres[0]
            centre_df = df_selected[df_selected['Centre'] == centre].copy()

            # Extract region and type for title
            region, tea_type = centre.split(' CTC ')

            # Sort by Sale No
            centre_df = centre_df.sort_values('Sale No')

            # Bar charts for Sold and Unsold
            fig.add_trace(go.Bar(name='Sold Qty (Ton)',
                                 x=centre_df['Sale No'],
                                 y=centre_df['Sold Qty (Ton)'],
                                 marker_color='#FF9966'),
                          row=1,
                          col=1,
                          secondary_y=False)

            fig.add_trace(go.Bar(name='Unsold Qty (Ton)',
                                 x=centre_df['Sale No'],
                                 y=centre_df['Unsold Qty (Ton)'],
                                 marker_color='#808080'),
                          row=1,
                          col=1,
                          secondary_y=False)

            # Line chart for Sales Price
            fig.add_trace(go.Scatter(name='Sales Price (Kg)',
                                     x=centre_df['Sale No'],
                                     y=centre_df['Sales Price(Kg)'],
                                     line=dict(color='#3366CC', width=2)),
                          row=1,
                          col=1,
                          secondary_y=True)

            # Add table trace with headers
            table_data = centre_df.sort_values('Sale No',
                                               ascending=True).copy()
            fig.add_trace(
                go.Table(
                    header=dict(values=[
                        'Metric',
                        *[f'Sale {x}' for x in table_data['Sale No'].tolist()]
                    ],
                               font=dict(size=11),
                               align='center',
                               height=30),
                    cells=dict(
                        values=[
                            ['Sold Qty', 'Unsold Qty',
                             'Price'],  # First column as metric names
                            *[[
                                table_data['Sold Qty (Ton)'].iloc[i],
                                table_data['Unsold Qty (Ton)'].iloc[i],
                                table_data['Sales Price(Kg)'].iloc[i]
                            ] for i in range(len(table_data))]
                        ],
                        font=dict(size=10),
                        align='center',
                        format=[None] + [',.0f'] * len(table_data),
                        height=25)),
                row=2,
                col=1)

            # Update layout for single chart
            fig.update_layout(
                title=f"{region} CTC {tea_type} Trends",
                height=800,
                barmode='group',
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=30, b=100, l=60,
                            r=60),  # Increased bottom margin
                showlegend=True)

            # Define Plotly config
            plotly_config = {
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["drawTools"],
                "showTips": True
            }

            # Render the chart with the updated config
            st.plotly_chart(fig,
                            use_container_width=True,
                            config=plotly_config)
        else:
            # Logic for multiple centres remains unchanged
            pass

        # AI-Powered Market Analysis section
        st.markdown("---")  # Add a visual separator
        st.markdown("""
            <h2 style='text-align: center; color: #1F4E79; margin-bottom: 2rem;'>AI-Powered Market Analysis</h2>
            
            <div style='display: flex; justify-content: space-between; gap: 2rem; margin-bottom: 2rem;'>
                <!-- Market Narrative -->
                <div style='flex: 1;'>
                    <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Market Narrative 📊</h3>
        """, unsafe_allow_html=True)
        
        if len(selected_centres) == 1:
            narrative = generate_ai_narrative(df_selected, selected_centres[0])
            st.markdown(narrative)
        else:
            st.info("Please select a single market for detailed AI analysis")
            
        st.markdown("""
                </div>
                
                <!-- Price Analysis -->
                <div style='flex: 1;'>
                    <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Price Analysis 💰</h3>
        """, unsafe_allow_html=True)
        
        if len(selected_centres) == 1:
            price_insights = generate_price_analysis(df_selected, selected_centres[0])
            st.markdown(price_insights)
        else:
            st.info("Please select a single market for price analysis")
            
        st.markdown("""
                </div>
                
                <!-- Market Insights -->
                <div style='flex: 1;'>
                    <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Market Insights 📈</h3>
        """, unsafe_allow_html=True)
        
        if len(selected_centres) == 1:
            market_insights = generate_market_insights(df_selected, selected_centres[0])
            st.markdown(market_insights)
        else:
            st.info("Please select a single market for market insights")
            
        st.markdown("""
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Statistical Analysis section
        st.markdown("---")  # Add a visual separator
        st.header("Statistical Analysis")
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.subheader("Market Position Analysis")
            if len(selected_centres) == 1:
                position_metric = st.selectbox(
                    "Select Position Metric",
                    ["Price", "Volume", "Efficiency"],
                    key="position_metric"
                )
                
                position_fig = go.Figure()
                centre_df = df_selected[df_selected['Centre'] == selected_centres[0]].copy()
                
                if position_metric == "Price":
                    y_data = centre_df['Sales Price(Kg)']
                    title = 'Price Levels Over Time'
                    y_title = 'Price (₹/Kg)'
                    metric_name = 'Price Levels'
                elif position_metric == "Volume":
                    y_data = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
                    title = 'Volume Levels Over Time'
                    y_title = 'Volume (Tons)'
                    metric_name = 'Volume Levels'
                else:  # Efficiency
                    y_data = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
                    title = 'Market Efficiency Over Time'
                    y_title = 'Efficiency Ratio'
                    metric_name = 'Efficiency Levels'
                
                position_fig.add_trace(go.Scatter(
                    x=centre_df['Sale No'],
                    y=y_data,
                    name=metric_name,
                    line=dict(color='#1F4E79', width=2)
                ))
                
                # Update layout
                position_fig.update_layout(
                    title=title,
                    xaxis_title='Sale Number',
                    yaxis_title=y_title,
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(position_fig, use_container_width=True)
                
                # Display insights
                levels_insights = analyze_levels(df_selected, selected_centres[0])
                for insight in levels_insights:
                    st.write(insight)
            else:
                st.info("Please select a single market for position analysis")
            
            st.subheader("Market Trends Analysis")
            if len(selected_centres) == 1:
                centre_df = df_selected[df_selected['Centre'] == selected_centres[0]].copy()
                centre_df = centre_df.sort_values('Sale No')
                
                # Price Trend Analysis
                price_fig = go.Figure()
                
                # Add actual price line
                price_fig.add_trace(
                    go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Sales Price(Kg)'],
                        name='Actual Price',
                        line=dict(color='#1F4E79', width=2)
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(centre_df)), centre_df['Sales Price(Kg)'], 1)
                p = np.poly1d(z)
                price_fig.add_trace(
                    go.Scatter(
                        x=centre_df['Sale No'],
                        y=p(range(len(centre_df))),
                        name='Trend Line',
                        line=dict(color='#FF9966', width=2, dash='dash')
                    )
                )
                
                # Update layout for price trend
                price_fig.update_layout(
                    title='Price Trend Analysis',
                    xaxis_title='Sale No',
                    yaxis_title='Price (₹/Kg)',
                    template='plotly_white',
                    height=300,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(price_fig, use_container_width=True)
                
                # Market Efficiency Trend
                trend_metric = st.selectbox(
                    "Select Trend Metric",
                    ["Sold/Total Ratio", "Price/Volume Correlation"],
                    key="trend_metric"
                )
                
                efficiency_fig = go.Figure()
                
                if trend_metric == "Sold/Total Ratio":
                    # Calculate efficiency ratio (Sold/Total)
                    centre_df['Efficiency'] = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
                    
                    efficiency_fig.add_trace(
                        go.Scatter(
                            x=centre_df['Sale No'],
                            y=centre_df['Efficiency'],
                            name='Sold/Total Ratio',
                            line=dict(color='#2E8B57', width=2)
                        )
                    )
                    
                    title = 'Market Efficiency - Sold/Total Ratio'
                    y_title = 'Ratio'
                    
                else:  # Price/Volume Correlation
                    # Calculate rolling correlation between price and volume
                    window = 3  # Use 3-sale window for correlation
                    centre_df['Total_Volume'] = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
                    centre_df['Price_Volume_Corr'] = centre_df['Sales Price(Kg)'].rolling(window=window).corr(centre_df['Total_Volume'])
                    
                    efficiency_fig.add_trace(
                        go.Scatter(
                            x=centre_df['Sale No'],
                            y=centre_df['Price_Volume_Corr'],
                            name='Price/Volume Correlation',
                            line=dict(color='#2E8B57', width=2)
                        )
                    )
                    
                    title = 'Market Efficiency - Price/Volume Correlation'
                    y_title = 'Correlation Coefficient'
                
                # Update layout for efficiency trend
                efficiency_fig.update_layout(
                    title=title,
                    xaxis_title='Sale No',
                    yaxis_title=y_title,
                    template='plotly_white',
                    height=300,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(efficiency_fig, use_container_width=True)
                
                # Display insights
                trends_insights = analyze_trends(df_selected, selected_centres[0])
                for insight in trends_insights:
                    st.write(insight)
            else:
                st.info("Please select a single market for trends analysis")
        
        with stat_col2:
            st.subheader("Comparative Analysis")
            if len(selected_centres) == 1:
                comparative_metric = st.selectbox(
                    "Select Comparison Metric",
                    ["Price", "Volume", "Efficiency"],
                    key="comparative_metric"
                )
                
                comparative_fig = go.Figure()
                region, tea_type = selected_centres[0].split(' CTC ')
                
                # Get all markets of same type
                similar_markets = [
                    market for market in df_selected['Centre'].unique()
                    if market.split(' CTC ')[1] == tea_type
                ]
                
                for market in similar_markets:
                    market_df = df_selected[df_selected['Centre'] == market].copy()
                    
                    if comparative_metric == "Price":
                        y_data = market_df['Sales Price(Kg)']
                        title = f'Price Comparison - {tea_type} Markets'
                        y_title = 'Price (₹/Kg)'
                    elif comparative_metric == "Volume":
                        y_data = market_df['Sold Qty (Ton)'] + market_df['Unsold Qty (Ton)']
                        title = f'Volume Comparison - {tea_type} Markets'
                        y_title = 'Volume (Tons)'
                    else:  # Efficiency
                        y_data = market_df['Sold Qty (Ton)'] / (market_df['Sold Qty (Ton)'] + market_df['Unsold Qty (Ton)'])
                        title = f'Efficiency Comparison - {tea_type} Markets'
                        y_title = 'Efficiency Ratio'
                    
                    comparative_fig.add_trace(go.Scatter(
                        x=market_df['Sale No'],
                        y=y_data,
                        name=market,
                        mode='lines'
                    ))
                
                # Update layout
                comparative_fig.update_layout(
                    title=title,
                    xaxis_title='Sale Number',
                    yaxis_title=y_title,
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(comparative_fig, use_container_width=True)
                
                # Display insights
                comparative_insights = analyze_comparatives(df_selected, selected_centres[0])
                for insight in comparative_insights:
                    st.write(insight)
            else:
                st.info("Please select a single market for comparative analysis")
            
            st.subheader("Correlation Analysis")
            if len(selected_centres) == 1:
                # Create correlation heatmap
                correlation_matrix = calculate_correlations(df_selected, selected_centres[0])
                correlation_fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                # Update layout
                correlation_fig.update_layout(
                    title='Metric Correlations',
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                # Display insights
                correlation_insights = analyze_key_correlations(df_selected, selected_centres[0])
                for insight in correlation_insights:
                    st.write(insight)
            else:
                st.info("Please select a single market for correlation analysis")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logging.error(f"Application error: {str(e)}")