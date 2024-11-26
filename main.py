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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logging.error(f"Application error: {str(e)}")
