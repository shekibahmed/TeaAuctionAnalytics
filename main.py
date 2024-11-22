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
                showlegend=True,
                modebar=dict(remove=[])  # Empty list to keep default modebar tools
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
                showlegend=True,
                modebar=dict(remove=[])  # Empty list to keep default modebar tools
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
            
            # Create tabs for different analyses
            tabs = st.tabs(["Trends", "Levels", "Comparatives", "Correlation Analysis"])
            
            # Trends Analysis Tab (now tabs[0])
            with tabs[0]:
                st.markdown("### Market Trends")
                trends_data = analyze_trends(df, centre)
                
                # Create expandable section for detailed trends analysis
                with st.expander("Click for Detailed Trends Analysis", expanded=False):
                    # Trend analysis period selector
                    st.markdown("#### Select Analysis Period")
                    trend_period = st.slider(
                        "Number of Sales for Trend Analysis",
                        min_value=3,
                        max_value=15,
                        value=5,
                        key=f"trend_period_{centre}"
                    )
                    
                    # Calculate and display trend metrics
                    centre_df = df[df['Centre'] == centre].sort_values('Sale No').tail(trend_period)
                    
                    # Price trend analysis
                    price_trend_fig = go.Figure()
                    
                    # Add actual price line
                    price_trend_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=centre_df['Sales Price(Kg)'],
                        name='Actual Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add trend line
                    z = np.polyfit(range(len(centre_df)), centre_df['Sales Price(Kg)'], 1)
                    p = np.poly1d(z)
                    price_trend_fig.add_trace(go.Scatter(
                        x=centre_df['Sale No'],
                        y=p(range(len(centre_df))),
                        name='Trend Line',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    price_trend_fig.update_layout(
                        title="Price Trend Analysis",
                        xaxis_title="Sale No",
                        yaxis_title="Price (₹/Kg)",
                        height=300,
                        modebar=dict(remove=[])  # Empty list to keep default modebar tools
                    )
                    st.plotly_chart(price_trend_fig, use_container_width=True)
                    
                    # Market efficiency trend
                    efficiency_option = st.selectbox(
                        "Efficiency Metric",
                        ["Sold/Total Ratio", "Price/Volume Correlation"],
                        key=f"efficiency_metric_{centre}"
                    )
                    
                    efficiency_fig = go.Figure()
                    
                    if efficiency_option == "Sold/Total Ratio":
                        efficiency = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
                        efficiency_fig.add_trace(go.Scatter(
                            x=centre_df['Sale No'],
                            y=efficiency,
                            name='Market Efficiency',
                            line=dict(color='green')
                        ))
                        efficiency_fig.update_layout(
                            title="Market Efficiency Trend",
                            xaxis_title="Sale No",
                            yaxis_title="Efficiency Ratio",
                            height=300,
                            modebar=dict(remove=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'])
                        )
                    else:
                        # Calculate rolling correlation
                        window = min(5, len(centre_df))
                        correlation = centre_df['Sales Price(Kg)'].rolling(window=window).corr(
                            centre_df['Sold Qty (Ton)']
                        )
                        efficiency_fig.add_trace(go.Scatter(
                            x=centre_df['Sale No'],
                            y=correlation,
                            name='Price-Volume Correlation',
                            line=dict(color='purple')
                        ))
                        efficiency_fig.update_layout(
                            title="Price-Volume Correlation Trend",
                            xaxis_title="Sale No",
                            yaxis_title="Correlation Coefficient",
                            height=300
                        )
                    
                    st.plotly_chart(efficiency_fig, use_container_width=True)
                
                # Display textual insights
                for insight in trends_data:
                    st.markdown(insight)
            
            # Levels Analysis Tab (now tabs[1])
            with tabs[1]:
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
                        title=f"Volume Analysis ({volume_agg if volume_agg != 'None' else 'Raw'})",
                        xaxis_title="Sale No",
                        yaxis_title="Volume (Tons)",
                        height=300
                    )
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                # Display textual insights
                for insight in levels_data:
                    st.markdown(insight)
            
            # Comparatives Analysis Tab (now tabs[2])
            with tabs[2]:
                st.markdown("### Market Comparatives")
                comparatives_data = analyze_comparatives(df, centre)
                
                # Create expandable section for detailed comparative analysis
                with st.expander("Click for Detailed Comparative Analysis", expanded=False):
                    # Metric selection for comparison
                    comparison_metric = st.selectbox(
                        "Select Comparison Metric",
                        ["Price", "Volume", "Efficiency"],
                        key=f"comparison_metric_{centre}"
                    )
                    
                    # Get comparison data
                    region, tea_type = centre.split(' CTC ')
                    other_type = 'Dust' if tea_type == 'Leaf' else 'Leaf'
                    other_centre = f"{region} CTC {other_type}"
                    
                    if other_centre in df['Centre'].unique():
                        centre_df = df[df['Centre'] == centre].sort_values('Sale No')
                        other_df = df[df['Centre'] == other_centre].sort_values('Sale No')
                        
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
                            comp_fig.update_layout(
                                title="Price Comparison",
                                xaxis_title="Sale No",
                                yaxis_title="Price (₹/Kg)",
                                height=300
                            )
                        elif comparison_metric == "Volume":
                            comp_fig.add_trace(go.Scatter(
                                x=centre_df['Sale No'],
                                y=centre_df['Sold Qty (Ton)'],
                                name=f'{tea_type} Volume',
                                line=dict(color='green')
                            ))
                            comp_fig.add_trace(go.Scatter(
                                x=other_df['Sale No'],
                                y=other_df['Sold Qty (Ton)'],
                                name=f'{other_type} Volume',
                                line=dict(color='orange')
                            ))
                            comp_fig.update_layout(
                                title="Volume Comparison",
                                xaxis_title="Sale No",
                                yaxis_title="Volume (Tons)",
                                height=300
                            )
                        else:  # Efficiency
                            centre_eff = centre_df['Sold Qty (Ton)'] / (centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)'])
                            other_eff = other_df['Sold Qty (Ton)'] / (other_df['Sold Qty (Ton)'] + other_df['Unsold Qty (Ton)'])
                            
                            comp_fig.add_trace(go.Scatter(
                                x=centre_df['Sale No'],
                                y=centre_eff,
                                name=f'{tea_type} Efficiency',
                                line=dict(color='purple')
                            ))
                            comp_fig.add_trace(go.Scatter(
                                x=other_df['Sale No'],
                                y=other_eff,
                                name=f'{other_type} Efficiency',
                                line=dict(color='brown')
                            ))
                            comp_fig.update_layout(
                                title="Efficiency Comparison",
                                xaxis_title="Sale No",
                                yaxis_title="Efficiency Ratio",
                                height=300
                            )
                        
                        st.plotly_chart(comp_fig, use_container_width=True)
                        
                        # Add statistical comparison
                        if comparison_metric == "Price":
                            avg_diff = centre_df['Sales Price(Kg)'].mean() - other_df['Sales Price(Kg)'].mean()
                            st.markdown(f"Average price difference: ₹{abs(avg_diff):.2f}/Kg ({'higher' if avg_diff > 0 else 'lower'})")
                        elif comparison_metric == "Volume":
                            avg_diff = centre_df['Sold Qty (Ton)'].mean() - other_df['Sold Qty (Ton)'].mean()
                            st.markdown(f"Average volume difference: {abs(avg_diff):.0f} tons ({'higher' if avg_diff > 0 else 'lower'})")
                        else:
                            centre_avg_eff = centre_eff.mean()
                            other_avg_eff = other_eff.mean()
                            st.markdown(f"Average efficiency: {centre_avg_eff*100:.1f}% vs {other_avg_eff*100:.1f}%")
                    
                    else:
                        st.warning(f"No comparison data available for {other_centre}")
                
                # Display textual insights
                for insight in comparatives_data:
                    st.markdown(insight)

            # Correlation Analysis Tab (now tabs[3])
            with tabs[3]:
                st.markdown("### Correlation Analysis")
                
                # Calculate correlation matrix
                corr_matrix = calculate_correlations(df, centre)
                
                # Create correlation heatmap
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                heatmap_fig.update_layout(
                    title="Metric Correlations Heatmap",
                    height=500,
                    width=700,
                    xaxis={'tickangle': 45}
                )
                
                st.plotly_chart(heatmap_fig)
                
                # Display key correlation insights
                st.markdown("#### Key Correlation Insights")
                correlation_insights = analyze_key_correlations(df, centre)
                for insight in correlation_insights:
                    st.markdown(insight)
                
                # Interactive Scatter Plot
                st.markdown("#### Interactive Correlation Scatter Plot")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_metric = st.selectbox(
                        "Select X-axis Metric",
                        options=corr_matrix.columns,
                        key=f"x_metric_{centre}"
                    )
                
                with col2:
                    y_metric = st.selectbox(
                        "Select Y-axis Metric",
                        options=[col for col in corr_matrix.columns if col != x_metric],
                        key=f"y_metric_{centre}"
                    )
                
                centre_df = df[df['Centre'] == centre].copy()
                centre_df['Total Volume'] = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
                centre_df['Market Efficiency'] = centre_df['Sold Qty (Ton)'] / centre_df['Total Volume']
                centre_df['Price Change'] = centre_df['Sales Price(Kg)'].pct_change()
                centre_df['Volume Change'] = centre_df['Total Volume'].pct_change()
                
                scatter_fig = go.Figure(data=go.Scatter(
                    x=centre_df[x_metric],
                    y=centre_df[y_metric],
                    mode='markers+text',
                    text=centre_df['Sale No'],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=centre_df['Sale No'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sale No")
                    )
                ))
                
                scatter_fig.update_layout(
                    title=f"Correlation: {x_metric} vs {y_metric}",
                    xaxis_title=x_metric,
                    yaxis_title=y_metric,
                    height=500
                )
                
                st.plotly_chart(scatter_fig, use_container_width=True)

        # Add download report button
        if st.button("Download PDF Report", key=f"download_report_{centre}"):
            pdf_report = generate_pdf_report(df, centre)
            st.download_button(
                "Click to Download",
                pdf_report,
                file_name=f"{centre}_market_analysis.pdf",
                mime="application/pdf"
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logging.error(f"Application error: {str(e)}")