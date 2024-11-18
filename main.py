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

        # AI-Powered Market Analysis Section (Now above Statistical Analysis)
        st.markdown("---")  # Add separator
        st.header("AI-Powered Market Analysis")
        
        for centre in selected_centres:
            st.subheader(f"{centre} Market Analysis")
            
            # Market Narrative in a row
            st.markdown("### 🤖 Market Narrative")
            narrative = generate_ai_narrative(df, centre)
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            {narrative}
            </div>
            """, unsafe_allow_html=True)
            
            # Price Analysis in a row
            st.markdown("### 📈 Price Analysis")
            price_analysis = generate_price_analysis(df, centre)
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            {price_analysis}
            </div>
            """, unsafe_allow_html=True)
            
            # Market Insights in a row
            st.markdown("### 🔍 Market Insights")
            market_insights = generate_market_insights(df, centre)
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            {market_insights}
            </div>
            """, unsafe_allow_html=True)

        # Statistical Analysis Section with Enhanced Interactive Drill-down
        st.markdown("---")  # Add separator
        st.header("Statistical Analysis")
        
        for centre in selected_centres:
            st.subheader(f"{centre} Statistical Analysis")
            
            # Create three columns for Levels, Trends, and Comparatives
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Price and Volume Levels")
                levels_data = analyze_levels(df, centre)
                
                # Create expandable section for detailed levels analysis with enhanced interactivity
                with st.expander("Click for Detailed Levels Analysis", expanded=False):
                    centre_df = df[df['Centre'] == centre].copy()
                    
                    # Add date range selector for levels analysis
                    st.markdown("#### Select Date Range for Analysis")
                    min_sale = int(centre_df['Sale No'].min())
                    max_sale = int(centre_df['Sale No'].max())
                    selected_range = st.slider(
                        "Sale Number Range",
                        min_value=min_sale,
                        max_value=max_sale,
                        value=(min_sale, max_sale),
                        key=f"levels_range_{centre}"
                    )
                    
                    # Filter data based on selection
                    filtered_df = centre_df[
                        (centre_df['Sale No'] >= selected_range[0]) &
                        (centre_df['Sale No'] <= selected_range[1])
                    ]
                    
                    # Price Distribution Chart with dynamic binning
                    num_bins = st.select_slider(
                        "Number of Price Distribution Bins",
                        options=[5, 10, 15, 20, 25, 30],
                        value=20,
                        key=f"price_bins_{centre}"
                    )
                    
                    price_dist_fig = go.Figure()
                    price_dist_fig.add_trace(go.Histogram(
                        x=filtered_df['Sales Price(Kg)'],
                        name='Price Distribution',
                        nbinsx=num_bins,
                        marker_color='blue'
                    ))
                    
                    latest_data = filtered_df.iloc[-1]
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
                    
                    # Volume Analysis with aggregation options
                    volume_agg = st.radio(
                        "Volume Aggregation",
                        ["None", "Moving Average", "Cumulative"],
                        key=f"volume_agg_{centre}"
                    )
                    
                    volume_fig = go.Figure()
                    
                    if volume_agg == "None":
                        y_sold = filtered_df['Sold Qty (Ton)']
                        y_unsold = filtered_df['Unsold Qty (Ton)']
                    elif volume_agg == "Moving Average":
                        window = st.slider("Moving Average Window", 2, 10, 5, key=f"ma_window_{centre}")
                        y_sold = filtered_df['Sold Qty (Ton)'].rolling(window=window).mean()
                        y_unsold = filtered_df['Unsold Qty (Ton)'].rolling(window=window).mean()
                    else:  # Cumulative
                        y_sold = filtered_df['Sold Qty (Ton)'].cumsum()
                        y_unsold = filtered_df['Unsold Qty (Ton)'].cumsum()
                    
                    volume_fig.add_trace(go.Scatter(
                        x=filtered_df['Sale No'],
                        y=y_sold,
                        name='Sold Volume',
                        line=dict(color='green')
                    ))
                    volume_fig.add_trace(go.Scatter(
                        x=filtered_df['Sale No'],
                        y=y_unsold,
                        name='Unsold Volume',
                        line=dict(color='red')
                    ))
                    
                    volume_fig.update_layout(
                        title=f"Volume Analysis Over Time ({volume_agg if volume_agg != 'None' else 'Raw Data'})",
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
                
                # Create expandable section for detailed trends analysis with enhanced interactivity
                with st.expander("Click for Detailed Trends Analysis", expanded=False):
                    # Allow users to select trend analysis type
                    trend_type = st.selectbox(
                        "Select Trend Analysis Type",
                        ["Price Trends", "Volume Trends", "Market Efficiency"],
                        key=f"trend_type_{centre}"
                    )
                    
                    # Allow users to customize trend line options
                    show_trend_line = st.checkbox("Show Trend Line", value=True, key=f"trend_line_{centre}")
                    show_ma = st.checkbox("Show Moving Average", value=True, key=f"ma_{centre}")
                    
                    if show_ma:
                        ma_window = st.slider(
                            "Moving Average Window",
                            min_value=2,
                            max_value=10,
                            value=5,
                            key=f"ma_window_trends_{centre}"
                        )
                    
                    trends_fig = go.Figure()
                    
                    if trend_type == "Price Trends":
                        y_data = filtered_df['Sales Price(Kg)']
                        title = "Price Trends Analysis"
                        y_label = "Price (₹/Kg)"
                    elif trend_type == "Volume Trends":
                        y_data = filtered_df['Sold Qty (Ton)']
                        title = "Volume Trends Analysis"
                        y_label = "Volume (Tons)"
                    else:  # Market Efficiency
                        y_data = filtered_df['Sold Qty (Ton)'] / (filtered_df['Sold Qty (Ton)'] + filtered_df['Unsold Qty (Ton)'])
                        title = "Market Efficiency Trends"
                        y_label = "Efficiency Ratio"
                    
                    # Add actual data line
                    trends_fig.add_trace(go.Scatter(
                        x=filtered_df['Sale No'],
                        y=y_data,
                        name='Actual Data',
                        line=dict(color='blue')
                    ))
                    
                    # Add trend line if selected
                    if show_trend_line:
                        z = np.polyfit(range(len(filtered_df)), y_data, 1)
                        p = np.poly1d(z)
                        trends_fig.add_trace(go.Scatter(
                            x=filtered_df['Sale No'],
                            y=p(range(len(filtered_df))),
                            name='Linear Trend',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    # Add moving average if selected
                    if show_ma:
                        moving_avg = y_data.rolling(window=ma_window).mean()
                        trends_fig.add_trace(go.Scatter(
                            x=filtered_df['Sale No'],
                            y=moving_avg,
                            name=f'{ma_window}-Sale Moving Average',
                            line=dict(color='green', dash='dot')
                        ))
                    
                    trends_fig.update_layout(
                        title=title,
                        xaxis_title="Sale No",
                        yaxis_title=y_label,
                        height=300
                    )
                    st.plotly_chart(trends_fig, use_container_width=True)
                    
                    # Display trend metrics
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
                        # Add comparison metric selector
                        comparison_metric = st.selectbox(
                            "Select Comparison Metric",
                            ["Price", "Volume", "Market Efficiency"],
                            key=f"comp_metric_{centre}"
                        )
                        
                        # Get data for both centres
                        centre_df = filtered_df
                        other_df = df[df['Centre'] == other_centre].copy()
                        other_df = other_df[
                            (other_df['Sale No'] >= selected_range[0]) &
                            (other_df['Sale No'] <= selected_range[1])
                        ]
                        
                        comp_fig = go.Figure()
                        
                        if comparison_metric == "Price":
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
                                line=dict(color='red')
                            ))
                            title = "Price Comparison"
                            y_label = "Price (₹/Kg)"
                            
                        elif comparison_metric == "Volume":
                            comp_fig.add_trace(go.Bar(
                                x=centre_df['Sale No'],
                                y=centre_df['Sold Qty (Ton)'],
                                name=f'{tea_type} Volume',
                                marker_color='blue'
                            ))
                            comp_fig.add_trace(go.Bar(
                                x=other_df['Sale No'],
                                y=other_df['Sold Qty (Ton)'],
                                name=f'{other_type} Volume',
                                marker_color='red'
                            ))
                            title = "Volume Comparison"
                            y_label = "Volume (Tons)"
                            
                        else:  # Market Efficiency
                            centre_eff = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
                            other_eff = other_df['Sold Qty (Ton)'] / (other_df['Sold Qty (Ton)'] + other_df['Unsold Qty (Ton)'])
                            
                            comp_fig.add_trace(go.Scatter(
                                x=centre_df['Sale No'],
                                y=centre_eff,
                                name=f'{tea_type} Efficiency',
                                line=dict(color='blue')
                            ))
                            comp_fig.add_trace(go.Scatter(
                                x=other_df['Sale No'],
                                y=other_eff,
                                name=f'{other_type} Efficiency',
                                line=dict(color='red')
                            ))
                            title = "Market Efficiency Comparison"
                            y_label = "Efficiency Ratio"
                        
                        comp_fig.update_layout(
                            title=title,
                            xaxis_title="Sale No",
                            yaxis_title=y_label,
                            height=300
                        )
                        st.plotly_chart(comp_fig, use_container_width=True)
                        
                        # Show detailed comparative metrics
                        st.markdown("\n".join(comparatives_data))
                    
                    else:
                        st.warning(f"No comparison data available for {other_centre}")

        # Download Report Section
        st.markdown("---")
        st.header("Download Analysis Report")
        
        # Allow user to select centres for report
        report_centres = st.multiselect(
            "Select Markets for Report",
            options=selected_centres,
            default=selected_centres[0] if selected_centres else None,
            key="report_centres"
        )
        
        if report_centres:
            if st.button("Generate PDF Report"):
                pdf_file = generate_pdf_report(df, report_centres)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name=f"{centre}_market_report.pdf",
                    mime="application/pdf"
                )

    else:
        # Show placeholders and instructions when no file is uploaded
        st.info("Please upload an Excel file to begin analysis.")
        st.markdown("""
        ### Expected File Format:
        The Excel file should contain the following columns:
        - Centre
        - Sale No
        - Sales Price(Kg)
        - Sold Qty (Ton)
        - Unsold Qty (Ton)
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logging.error(f"Error in main app: {str(e)}")