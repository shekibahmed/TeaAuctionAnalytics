import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import (process_excel_data, generate_price_analysis,
                   generate_market_insights, generate_volume_analysis,
                   generate_recommendations, generate_ai_narrative,
                   analyze_levels, analyze_trends, analyze_comparatives, generate_pdf_report)
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

            # Add table trace with headers
            table_data = centre_df.sort_values('Sale No', ascending=True).copy()
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Metric', *[f'Sale {x}' for x in table_data['Sale No'].tolist()]],
                        font=dict(size=11),
                        align='center',
                        height=30
                    ),
                    cells=dict(
                        values=[
                            ['Sold Qty', 'Unsold Qty', 'Price'],  # First column as metric names
                            *[
                                [
                                    table_data['Sold Qty (Ton)'].iloc[i],
                                    table_data['Unsold Qty (Ton)'].iloc[i],
                                    table_data['Sales Price(Kg)'].iloc[i]
                                ] for i in range(len(table_data))
                            ]
                        ],
                        font=dict(size=10),
                        align='center',
                        format=[None] + [',.0f'] * len(table_data),
                        height=25
                    )
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
                margin=dict(t=30, b=100, l=60, r=60),  # Increased bottom margin
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
                vertical_spacing=0.1,  # Increased spacing
                row_heights=[0.7, 0.3] * rows  # Adjusted ratio between chart and table
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

                # Add table trace with headers
                table_data = centre_df.sort_values('Sale No', ascending=True).copy()
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Metric', *[f'Sale {x}' for x in table_data['Sale No'].tolist()]],
                            font=dict(size=11),
                            align='center',
                            height=30
                        ),
                        cells=dict(
                            values=[
                                ['Sold Qty', 'Unsold Qty', 'Price'],  # First column as metric names
                                *[
                                    [
                                        table_data['Sold Qty (Ton)'].iloc[i],
                                        table_data['Unsold Qty (Ton)'].iloc[i],
                                        table_data['Sales Price(Kg)'].iloc[i]
                                    ] for i in range(len(table_data))
                                ]
                            ],
                            font=dict(size=9),
                            align='center',
                            format=[None] + [',.0f'] * len(table_data),
                            height=25
                        )
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
                margin=dict(t=50, b=100, l=60, r=60),  # Increased bottom margin
                showlegend=True
            )

        # Display charts with integrated tables
        st.plotly_chart(fig, use_container_width=True)

        # Statistical Analysis Section with Interactive Drill-down
        st.header("Statistical Analysis")
        
        for centre in selected_centres:
            st.subheader(f"{centre} Statistical Analysis")
            
            # Create three columns for Levels, Trends, and Comparatives
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Price and Volume Levels")
                levels_data = analyze_levels(df, centre)
                
                # Create expandable section for detailed levels analysis
                with st.expander("Click for Detailed Levels Analysis", expanded=False):
                    centre_df = df[df['Centre'] == centre].copy()
                    latest_sale = centre_df['Sale No'].max()
                    latest_data = centre_df[centre_df['Sale No'] == latest_sale].iloc[0]
                    
                    # Price Distribution Chart
                    price_dist_fig = go.Figure()
                    price_dist_fig.add_trace(go.Histogram(
                        x=centre_df['Sales Price(Kg)'],
                        name='Price Distribution',
                        nbinsx=20,
                        marker_color='blue'
                    ))
                    price_dist_fig.add_vline(
                        x=latest_data['Sales Price(Kg)'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Current Price"
                    )
                    price_dist_fig.update_layout(
                        title="Price Distribution Analysis",
                        xaxis_title="Price (₹/Kg)",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(price_dist_fig, use_container_width=True)
                    
                    # Volume Analysis
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Sold Qty (Ton)'],
                        name='Sold Volume',
                        line=dict(color='green')
                    ))
                    volume_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Unsold Qty (Ton)'],
                        name='Unsold Volume',
                        line=dict(color='red')
                    ))
                    volume_fig.update_layout(
                        title="Volume Analysis Over Time",
                        xaxis_title="Sale No",
                        yaxis_title="Volume (Tons)",
                        height=300
                    )
                    st.plotly_chart(volume_fig, use_container_width=True)
                    
                    # Detailed metrics
                    st.markdown("\n".join(levels_data))
            
            with col2:
                st.markdown("### Market Trends")
                trends_data = analyze_trends(df, centre)
                
                # Create expandable section for detailed trends analysis
                with st.expander("Click for Detailed Trends Analysis", expanded=False):
                    centre_df = centre_df.sort_values('Sale No')
                    
                    # Price Trend Analysis
                    trends_fig = go.Figure()
                    
                    # Add actual price line
                    trends_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Sales Price(Kg)'],
                        name='Actual Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add trend line
                    z = np.polyfit(range(len(centre_df)), centre_df['Sales Price(Kg)'], 1)
                    p = np.poly1d(z)
                    trends_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=p(range(len(centre_df))),
                        name='Linear Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add moving average
                    ma_window = min(5, len(centre_df))
                    moving_avg = centre_df['Sales Price(Kg)'].rolling(window=ma_window).mean()
                    trends_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=moving_avg,
                        name=f'{ma_window}-Sale Moving Average',
                        line=dict(color='green', dash='dot')
                    ))
                    
                    trends_fig.update_layout(
                        title="Detailed Price Trend Analysis",
                        xaxis_title="Sale No",
                        yaxis_title="Price (₹/Kg)",
                        height=300
                    )
                    st.plotly_chart(trends_fig, use_container_width=True)
                    
                    # Volume Trend Analysis
                    volume_trends_fig = go.Figure()
                    volume_trends_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']),
                        name='Market Efficiency',
                        line=dict(color='purple')
                    ))
                    volume_trends_fig.update_layout(
                        title="Market Efficiency Trend",
                        xaxis_title="Sale No",
                        yaxis_title="Efficiency Ratio",
                        height=300
                    )
                    st.plotly_chart(volume_trends_fig, use_container_width=True)
                    
                    # Detailed metrics
                    st.markdown("\n".join(trends_data))
            
            with col3:
                st.markdown("### Market Comparatives")
                comparatives_data = analyze_comparatives(df, centre)
                
                # Create expandable section for detailed comparatives analysis
                with st.expander("Click for Detailed Comparatives Analysis", expanded=False):
                    region, tea_type = centre.split(' CTC ')
                    other_type = 'Dust' if tea_type == 'Leaf' else 'Leaf'
                    other_centre = f"{region} CTC {other_type}"
                    
                    if other_centre in df['Centre'].unique():
                        # Comparative Price Analysis
                        centre_df = df[df['Centre'] == centre].copy()
                        other_df = df[df['Centre'] == other_centre].copy()
                        
                        comp_fig = go.Figure()
                        
                        # Add price lines
                        comp_fig.add_trace(go.Scatter(
                            x=centre_df['Sale No'],
                            y=centre_df['Sales Price(Kg)'],
                            name=f'{tea_type} Price',
                            line=dict(color='blue')
                        ))
                        
                        comp_fig.add_trace(go.Scatter(
                            x=other_df['Sale No'],
                            y=other_df['Sales Price(Kg)'],
                            name=f'{other_type} Price',
                            line=dict(color='green')
                        ))
                        
                        # Add price ratio
                        price_ratio = pd.merge(
                            centre_df[['Sale No', 'Sales Price(Kg)']].rename(columns={'Sales Price(Kg)': 'price1'}),
                            other_df[['Sale No', 'Sales Price(Kg)']].rename(columns={'Sales Price(Kg)': 'price2'}),
                            on='Sale No'
                        )
                        price_ratio['ratio'] = price_ratio['price1'] / price_ratio['price2']
                        
                        comp_fig.add_trace(go.Scatter(
                            x=price_ratio['Sale No'],
                            y=price_ratio['ratio'],
                            name='Price Ratio',
                            line=dict(color='red', dash='dash'),
                            yaxis='y2'
                        ))
                        
                        comp_fig.update_layout(
                            title="Comparative Price Analysis",
                            xaxis_title="Sale No",
                            yaxis_title="Price (₹/Kg)",
                            yaxis2=dict(
                                title="Price Ratio",
                                overlaying="y",
                                side="right"
                            ),
                            height=300
                        )
                        st.plotly_chart(comp_fig, use_container_width=True)
                        
                        # Market Share Analysis
                        share_fig = go.Figure()
                        total_volume = centre_df['Sold Qty (Ton)'] + other_df['Sold Qty (Ton)']
                        share_fig.add_trace(go.Bar(
                            x=centre_df['Sale No'],
                            y=centre_df['Sold Qty (Ton)'] / total_volume * 100,
                            name=f'{tea_type} Share',
                            marker_color='blue'
                        ))
                        
                        share_fig.update_layout(
                            title="Market Share Analysis",
                            xaxis_title="Sale No",
                            yaxis_title="Market Share (%)",
                            height=300
                        )
                        st.plotly_chart(share_fig, use_container_width=True)
                    
                    # Detailed metrics
                    st.markdown("\n".join(comparatives_data))
            
            # Add Download PDF Report Button
            st.markdown("### Download Statistical Report")
            if st.button(f"Generate PDF Report for {centre}"):
                pdf_data = generate_pdf_report(df, centre)
                st.download_button(
                    label=f"Download {centre} Report",
                    data=pdf_data,
                    file_name=f"{centre}_market_report.pdf",
                    mime="application/pdf"
                )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")