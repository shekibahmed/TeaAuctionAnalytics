import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import (process_excel_data, generate_price_analysis,
                   generate_market_insights, generate_volume_analysis,
                   generate_recommendations, generate_ai_narrative)
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
        regions = sorted(list(set([centre.split(' CTC ')[0] for centre in df['Centre'].unique()])))
        tea_types = sorted(list(set([centre.split(' CTC ')[1] for centre in df['Centre'].unique()])))

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
                specs=[[{"secondary_y": True}], [{"type": "table"}]],
                row_heights=[0.85, 0.15],
                vertical_spacing=0.02
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
                       row=1, col=1, secondary_y=False)

            fig.add_trace(go.Bar(name='Unsold Qty (Ton)',
                              x=centre_df['Sale No'],
                              y=centre_df['Unsold Qty (Ton)'],
                              marker_color='#808080'),
                       row=1, col=1, secondary_y=False)

            # Line chart for Sales Price
            fig.add_trace(go.Scatter(name='Sales Price (Kg)',
                                  x=centre_df['Sale No'],
                                  y=centre_df['Sales Price(Kg)'],
                                  line=dict(color='#3366CC', width=2)),
                       row=1, col=1, secondary_y=True)

            # Add table trace
            table_data = centre_df.sort_values('Sale No', ascending=True).copy()
            fig.add_trace(
                go.Table(
                    cells=dict(
                        values=[
                            ['Sale No'] + table_data['Sale No'].tolist(),
                            ['Sold Qty'] + table_data['Sold Qty (Ton)'].round(0).astype(int).tolist(),
                            ['Unsold Qty'] + table_data['Unsold Qty (Ton)'].round(0).astype(int).tolist(),
                            ['Price'] + table_data['Sales Price(Kg)'].round(0).astype(int).tolist()
                        ],
                        font=dict(size=10),
                        align=['center'] * 4,
                        format=[None, 'd', 'd', 'd'],
                        height=25,
                        line=dict(width=0)
                    ),
                    columnwidth=[1, 1, 1, 1]
                ),
                row=2, col=1
            )

            # Update layout for single chart
            fig.update_layout(
                title=f"{region} CTC {tea_type} Trends",
                height=800,
                barmode='group',
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=30, b=50, l=60, r=60),  # Increased bottom margin
                showlegend=True
            )

            fig.update_xaxes(title_text='Sale No', row=1, col=1, showgrid=True)
            fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False, row=1, col=1)
            fig.update_yaxes(title_text='Price (₹/Kg)', secondary_y=True, row=1, col=1)

        else:
            # Multiple centers - grid layout with tables
            rows = (num_centres + 1) // 2
            fig = make_subplots(
                rows=rows * 2,
                cols=min(2, num_centres),
                specs=[[{"secondary_y": True}, {"secondary_y": True}] if i % 2 == 0 else [{"type": "table"}, {"type": "table"}] for i in range(rows * 2)],
                subplot_titles=[
                    f"{centre.split(' CTC ')[0]} CTC {centre.split(' CTC ')[1]} Trends"
                    for centre in selected_centres
                ],
                vertical_spacing=0.02,
                row_heights=[0.4, 0.1] * rows  # Adjusted ratio between chart and table
            )

            for idx, centre in enumerate(selected_centres):
                chart_row = (idx // 2) * 2 + 1
                table_row = chart_row + 1
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
                    row=chart_row, col=col, secondary_y=False
                )

                fig.add_trace(
                    go.Bar(
                        name='Unsold Qty (Ton)',
                        x=centre_df['Sale No'],
                        y=centre_df['Unsold Qty (Ton)'],
                        marker_color='#808080',
                        showlegend=(idx == 0)
                    ),
                    row=chart_row, col=col, secondary_y=False
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
                    row=chart_row, col=col, secondary_y=True
                )

                # Add table trace
                table_data = centre_df.sort_values('Sale No', ascending=True).copy()
                fig.add_trace(
                    go.Table(
                        cells=dict(
                            values=[
                                ['Sale No'] + table_data['Sale No'].tolist(),
                                ['Sold Qty'] + table_data['Sold Qty (Ton)'].round(0).astype(int).tolist(),
                                ['Unsold Qty'] + table_data['Unsold Qty (Ton)'].round(0).astype(int).tolist(),
                                ['Price'] + table_data['Sales Price(Kg)'].round(0).astype(int).tolist()
                            ],
                            font=dict(size=9),
                            align=['center'] * 4,
                            format=[None, 'd', 'd', 'd'],
                            height=25,
                            line=dict(width=0)
                        ),
                        columnwidth=[1, 1, 1, 1]
                    ),
                    row=table_row, col=col
                )

                # Update axes labels
                fig.update_xaxes(title_text='Sale No', row=chart_row, col=col, showgrid=True)
                fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False, row=chart_row, col=col)
                fig.update_yaxes(title_text='Price (₹/Kg)', secondary_y=True, row=chart_row, col=col)

            # Update layout for multiple charts
            fig.update_layout(
                height=500 * rows,
                barmode='group',
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=50, b=50, l=60, r=60),  # Increased bottom margin
                showlegend=True
            )

        # Display charts with integrated tables
        st.plotly_chart(fig, use_container_width=True)

        # Summary Metrics
        st.subheader("Key Metrics by Market")
        metrics_cols = st.columns(len(selected_centres))

        for idx, (col, centre) in enumerate(zip(metrics_cols, selected_centres)):
            centre_df = df_selected[df_selected['Centre'] == centre].copy()
            region, tea_type = centre.split(' CTC ')

            with col:
                st.markdown(f"**{region} CTC {tea_type}**")

                avg_price = centre_df['Sales Price(Kg)'].mean()
                price_change = centre_df['Sales Price(Kg)'].pct_change().mean()
                total_sold = centre_df['Sold Qty (Ton)'].sum()
                sold_change = centre_df['Sold Qty (Ton)'].pct_change().mean()
                total_qty = total_sold + centre_df['Unsold Qty (Ton)'].sum()
                efficiency = (total_sold / total_qty * 100) if total_qty > 0 else 0

                st.metric(
                    "Avg Sales Price", f"₹{avg_price:.2f}/Kg",
                    f"{price_change*100:.1f}%" if pd.notna(price_change) else None)
                st.metric(
                    "Total Sold Qty", f"{total_sold:,.0f} Tons",
                    f"{sold_change*100:.1f}%" if pd.notna(sold_change) else None)
                st.metric("Market Efficiency", f"{efficiency:.1f}%")

        # Market Insights with Expandable Sections
        st.header("Market Insights")
        st.markdown("""
        Click on each section below to view detailed market analysis:
        """)

        for centre in selected_centres:
            st.subheader(f"{centre} Analysis")

            # AI-powered Narrative Analysis Section
            with st.expander("🤖 AI Market Analysis", expanded=True):
                narrative = generate_ai_narrative(df, centre)
                st.markdown(narrative)

            # Price Analysis Section
            with st.expander("🏷️ Price Analysis", expanded=False):
                price_analysis = generate_price_analysis(df, centre)
                st.markdown(price_analysis)

            # Market Insights Section
            with st.expander("📊 Market Insights", expanded=False):
                market_analysis = generate_market_insights(df, centre)
                st.markdown(market_analysis)

            # Volume Analysis Section
            with st.expander("📈 Volume Analysis", expanded=False):
                volume_analysis = generate_volume_analysis(df, centre)
                st.markdown(volume_analysis)

            # Recommendations Section
            with st.expander("💡 Strategic Recommendations", expanded=False):
                recommendations = generate_recommendations(df, centre)
                st.markdown(recommendations)

    else:
        # Show placeholders and instructions when no file is uploaded
        st.info("Upload a file above to start analyzing your tea market data")

        # Placeholder for charts
        st.header("Market Trends")
        st.markdown("Charts will appear here after uploading data")

        # Placeholder for metrics
        st.header("Key Metrics by Market")
        st.markdown("Market metrics will be displayed here after data upload")

        # Placeholder for insights
        st.header("Market Insights")
        st.markdown(
            "AI-powered market analysis will be generated here after data upload"
        )

except Exception as e:
    st.error("""
    An error occurred while processing the data. Please ensure:
    1. The file format is correct (.xlsx, .xls, or .csv)
    2. All required columns are present
    3. Market categories follow the format: "[Region] CTC [Type]"
    4. All numeric values are valid
    
    Error details: """ + str(e))

    if "Invalid market categories" in str(e):
        st.info("""
        Market categories must follow the format: "[Region] CTC [Type]"
        - Region must be either "North India" or "South India"
        - Type must be either "Leaf" or "Dust"
        
        Examples:
        - North India CTC Leaf
        - South India CTC Dust
        """)
    elif "Missing required columns" in str(e):
        st.info("""
        Please ensure your file contains all required columns:
        - Centre (Market location)
        - Sale No (Sale number)
        - Sales Price(Kg) (Price per kilogram)
        - Sold Qty (Ton) (Quantity sold in tons)
        - Unsold Qty (Ton) (Unsold quantity in tons)
        """)
