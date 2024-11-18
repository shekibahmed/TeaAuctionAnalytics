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
                        height=300
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
                            height=300
                        )
                    else:
                        # Calculate rolling correlation
                        window = min(5, len(centre_df))
                        total_vol = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
                        corr = pd.DataFrame({
                            'Price': centre_df['Sales Price(Kg)'],
                            'Volume': total_vol
                        }).rolling(window).corr().iloc[::2]['Volume']
                        
                        efficiency_fig.add_trace(go.Scatter(
                            x=centre_df['Sale No'],
                            y=corr,
                            name='Price-Volume Correlation',
                            line=dict(color='purple')
                        ))
                        efficiency_fig.update_layout(
                            title="Price-Volume Correlation Trend",
                            xaxis_title="Sale No",
                            yaxis_title="Correlation Coefficient",
                            height=300
                        )
                    
                    st.plotly_chart(
                        efficiency_fig,
                        use_container_width=True,
                        config={'responsive': True, 'displayModeBar': False}
                    )
            
            # Levels Analysis Tab (now tabs[1])
            with tabs[1]:
                st.markdown("### Market Levels")
                levels_data = analyze_levels(df, centre)
                for insight in levels_data:
                    st.write(insight)
            
            # Comparatives Analysis Tab (now tabs[2])
            with tabs[2]:
                st.markdown("### Market Comparatives")
                comparative_data = analyze_comparatives(df, centre)
                for insight in comparative_data:
                    st.write(insight)
            
            # Correlation Analysis Tab (now tabs[3])
            with tabs[3]:
                st.markdown("### Correlation Analysis")
                
                with st.container():
                    # Calculate correlation matrix
                    corr_matrix = calculate_correlations(df, centre)
                    
                    # Create heatmap
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    
                    heatmap_fig.update_layout(
                        title="Correlation Matrix Heatmap",
                        height=400,
                        width=600,
                        autosize=True,
                        margin=dict(t=50, l=50, r=50, b=50)
                    )
                    
                    st.plotly_chart(
                        heatmap_fig,
                        use_container_width=True,
                        config={'responsive': True, 'displayModeBar': False}
                    )
                    
                    # Display correlation insights
                    st.markdown("#### Key Correlation Insights")
                    correlation_insights = analyze_key_correlations(df, centre)
                    for insight in correlation_insights:
                        st.write(insight)
                    
                    # Add scatter plots for key relationships
                    scatter_container = st.container()
                    with scatter_container:
                        centre_df = df[df['Centre'] == centre].copy()
                        centre_df['Total Volume'] = centre_df['Sold Qty (Ton)'] + centre_df['Unsold Qty (Ton)']
                        centre_df['Market Efficiency'] = centre_df['Sold Qty (Ton)'] / centre_df['Total Volume']
                        
                        # Create scatter plot
                        scatter_fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Price vs Volume", "Price vs Market Efficiency")
                        )
                        
                        scatter_fig.add_trace(
                            go.Scatter(
                                x=centre_df['Total Volume'],
                                y=centre_df['Sales Price(Kg)'],
                                mode='markers',
                                name='Price vs Volume',
                                marker=dict(color='blue', size=8)
                            ),
                            row=1, col=1
                        )
                        
                        scatter_fig.add_trace(
                            go.Scatter(
                                x=centre_df['Market Efficiency'],
                                y=centre_df['Sales Price(Kg)'],
                                mode='markers',
                                name='Price vs Efficiency',
                                marker=dict(color='green', size=8)
                            ),
                            row=1, col=2
                        )
                        
                        scatter_fig.update_layout(
                            height=400,
                            autosize=True,
                            showlegend=False,
                            margin=dict(t=50, l=50, r=50, b=50)
                        )
                        
                        scatter_fig.update_xaxes(title_text="Total Volume (Tons)", row=1, col=1)
                        scatter_fig.update_xaxes(title_text="Market Efficiency", row=1, col=2)
                        scatter_fig.update_yaxes(title_text="Price (₹/Kg)", row=1, col=1)
                        scatter_fig.update_yaxes(title_text="Price (₹/Kg)", row=1, col=2)
                        
                        st.plotly_chart(
                            scatter_fig,
                            use_container_width=True,
                            config={'responsive': True, 'displayModeBar': False}
                        )

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