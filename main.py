import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("Importing required packages...")
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import numpy as np
    from utils import (process_excel_data, generate_price_analysis,
                       generate_market_insights, generate_ai_narrative,
                       analyze_levels, analyze_trends, analyze_comparatives,
                       calculate_correlations, analyze_key_correlations)
    from styles import apply_custom_styles
    from loading_animations import (TeaLoadingAnimations, ProgressTracker,
                                    show_loading_animation,
                                    simulate_processing_delay,
                                    show_advanced_loading_animation,
                                    EnhancedProgressTracker)
    import os

    # Setup page config
    st.set_page_config(page_title="CTC Tea Sales Analytics Dashboard",
                       layout="wide")
    apply_custom_styles()

    # Welcome animation for first time users
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        welcome_stages = [
            "Warming up the tea brewing station...",
            "Selecting finest CTC leaves for analysis...",
            "Preparing dashboard with market intelligence...",
            "Ready to serve your analytics!"
        ]
        show_loading_animation("brewing_process", stages=welcome_stages)

    # Add title with description
    st.title("Auction Analytics")
    st.markdown("""
    This dashboard provides comprehensive analysis of CTC tea sales across North and South India markets,
    featuring AI-powered market insights and traditional market metrics.
    """)

    try:
        # File upload section with sample data option
        st.subheader("📁 Data Input")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload Your Data File",
                type=['xlsx', 'xls', 'csv'],
                help="Upload Excel or CSV files with your CTC tea auction data"
            )

        with col2:
            st.markdown("**Or try with sample data:**")
            use_sample = st.button(
                "Load Sample Data",
                help="Load demo data to explore dashboard features",
                type="secondary")

            if st.button("ℹ️ About Sample Data"):
                st.info("""
                **Sample Dataset Overview:**
                - 24 records from North & South India CTC markets
                - Both Leaf and Dust tea varieties included  
                - Sale numbers 40-46 (2024) representing Guwahati auction sessions
                - Price, volume, and unsold quantity metrics
                - Perfect for testing all dashboard features
                """)

                # Show a preview of the sample data
                try:
                    sample_preview = pd.read_csv('assets/default_data.csv')
                    st.subheader("📊 Data Preview")
                    st.dataframe(sample_preview.head(8),
                                 use_container_width=True)
                    st.caption(
                        f"Showing first 8 of {len(sample_preview)} total records"
                    )
                except Exception as e:
                    st.error(f"Could not load preview: {str(e)}")

        # Process data based on user choice
        df = None
        if uploaded_file is not None:
            # Show animated loading for file upload
            with ProgressTracker(3, 'upload') as progress:
                progress.update("Reading file contents...")
                simulate_processing_delay(0.3, 0.8)

                progress.update("Validating data structure...")
                simulate_processing_delay(0.2, 0.6)

                progress.update("Processing Excel data...")
                df = process_excel_data(uploaded_file)

            st.success("✅ File uploaded and processed successfully!")

        elif use_sample:
            try:
                # Show tea garden growing animation for sample data
                show_loading_animation(
                    "tea_garden",
                    message="Loading sample CTC tea market data...")

                # Load sample data and process it through the same pipeline
                import io
                with open('assets/default_data.csv', 'rb') as f:
                    sample_data = io.BytesIO(f.read())
                sample_data.name = 'default_data.csv'  # Set name for processing
                df = process_excel_data(sample_data)

                st.success("✅ Sample data loaded successfully!")
                st.info(
                    "📊 This sample contains CTC tea auction data from North and South India markets for demonstration purposes."
                )
            except Exception as e:
                st.error(f"❌ Error loading sample data: {str(e)}")
                df = None

        if df is not None:

            # Center selection with region and type filtering
            df = df.copy()  # Ensure we have a proper DataFrame copy
            centres_list = df['Centre'].tolist()
            regions = sorted(
                list(set([centre.split(' CTC ')[0]
                          for centre in centres_list])))
            tea_types = sorted(
                list(set([centre.split(' CTC ')[1]
                          for centre in centres_list])))

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

            # Filter centres based on region and type selection (remove duplicates)
            selected_centres = sorted(
                list(
                    set([
                        centre for centre in centres_list
                        if any(region in centre
                               for region in selected_regions) and any(
                                   tea_type in centre
                                   for tea_type in selected_types)
                    ])))

            if not selected_centres:
                st.warning("Please select at least one region and tea type.")
                st.stop()

            # Filter data for selected centres and ensure proper DataFrame type
            df_selected = df[df['Centre'].isin(selected_centres)].copy()
            df_selected = pd.DataFrame(
                df_selected)  # Ensure proper DataFrame type

            # Create charts based on selection
            num_centres = len(selected_centres)
            if num_centres == 1:
                # Single center - larger chart with table
                fig = make_subplots(rows=2,
                                    cols=1,
                                    specs=[[{
                                        "secondary_y": True
                                    }], [{
                                        "type": "table"
                                    }]],
                                    row_heights=[0.7, 0.3],
                                    vertical_spacing=0.1)

                centre = selected_centres[0]
                centre_df = df_selected[df_selected['Centre'] == centre].copy()
                centre_df = pd.DataFrame(centre_df)  # Ensure DataFrame type

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
                            'Metric', *[
                                f'Sale {x}'
                                for x in table_data['Sale No'].tolist()
                            ]
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
                fig.update_layout(title=f"{region} CTC {tea_type} Trends",
                                  height=800,
                                  barmode='group',
                                  hovermode='x unified',
                                  template='plotly_white',
                                  margin=dict(t=30, b=100, l=60, r=60),
                                  showlegend=True)

                # Define Plotly config with ResizeObserver optimization
                plotly_config = {
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["drawTools"],
                    "showTips": True,
                    "responsive": True,
                    "staticPlot": False
                }

                # Render the chart with optimized config
                st.plotly_chart(fig,
                                use_container_width=True,
                                config=plotly_config,
                                key="main_chart")

            # AI-Powered Market Analysis section
            st.markdown("---")  # Add a visual separator
            st.markdown("""
                <h2 style='text-align: center; color: #1F4E79; margin-bottom: 2rem;'>AI-Powered Market Analysis</h2>
                
                <div style='display: flex; justify-content: space-between; gap: 2rem; margin-bottom: 2rem;'>
                    <!-- Market Narrative -->
                    <div style='flex: 1;'>
                        <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Market Narrative 📊</h3>
            """,
                        unsafe_allow_html=True)

            if len(selected_centres) == 1:
                with ProgressTracker(2, 'ai_analysis') as progress:
                    progress.update("AI sommelier analyzing your data...")
                    simulate_processing_delay(0.5, 1.0)

                    progress.update("Crafting market narrative...")
                    narrative = generate_ai_narrative(df_selected,
                                                      selected_centres[0])
                st.markdown(narrative)
            else:
                st.info(
                    "Please select a single market for detailed AI analysis")

            st.markdown("""
                    </div>
                    
                    <!-- Price Analysis -->
                    <div style='flex: 1;'>
                        <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Price Analysis 💰</h3>
            """,
                        unsafe_allow_html=True)

            if len(selected_centres) == 1:
                with ProgressTracker(2, 'ai_analysis') as progress:
                    progress.update("Analyzing price patterns and trends...")
                    simulate_processing_delay(0.4, 0.8)

                    progress.update("Generating price intelligence...")
                    price_insights = generate_price_analysis(
                        df_selected, selected_centres[0])
                st.markdown(price_insights)
            else:
                st.info("Please select a single market for price analysis")

            st.markdown("""
                    </div>
                    
                    <!-- Market Insights -->
                    <div style='flex: 1;'>
                        <h3 style='color: #1F4E79; margin-bottom: 1rem;'>Market Insights 📈</h3>
            """,
                        unsafe_allow_html=True)

            if len(selected_centres) == 1:
                with ProgressTracker(2, 'ai_analysis') as progress:
                    progress.update("Distilling market wisdom...")
                    simulate_processing_delay(0.3, 0.7)

                    progress.update("Crafting strategic insights...")
                    market_insights = generate_market_insights(
                        df_selected, selected_centres[0])
                st.markdown(market_insights)
            else:
                st.info("Please select a single market for market insights")

            st.markdown("""
                    </div>
                </div>
            """,
                        unsafe_allow_html=True)

            # Statistical Analysis section with tabs
            st.markdown("---")  # Add a visual separator
            st.header("Statistical Analysis")

            # Enhanced tabs with touch-friendly icons and responsive layout
            tab_styles = """
                <style>
                    .tab-icon { font-size: 1.2em; margin-right: 8px; }
                    @media (max-width: 768px) {
                        .tab-icon { display: block; margin: 0 auto 4px; }
                    }
                </style>
            """
            st.markdown(tab_styles, unsafe_allow_html=True)

            tabs = st.tabs(["📊 Position", "📈 Trends", "🔄 Compare", "💰 Levels"])

            # Market Position Analysis Tab
            with tabs[0]:  # Market Position
                if len(selected_centres) == 1:
                    # Controls Section - Mobile Optimized
                    with st.expander("📊 Controls", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            position_metric = st.selectbox(
                                "Metric", ["Price", "Volume", "Efficiency"],
                                key="position_metric",
                                help="Choose the metric to analyze")
                        with col2:
                            date_range = st.slider(
                                "Period",
                                min_value=1,
                                max_value=30,
                                value=(1, 30),
                                key="position_date_range",
                                help="Select analysis time period")

                    # Data Processing with animation
                    with ProgressTracker(3, 'processing') as progress:
                        progress.update(
                            "Filtering data for selected centre...")
                        centre_df = df_selected[df_selected['Centre'] ==
                                                selected_centres[0]].copy()
                        simulate_processing_delay(0.2, 0.5)

                        progress.update(
                            "Calculating market position metrics...")
                        simulate_processing_delay(0.3, 0.6)

                        progress.update("Preparing visualization data...")
                        simulate_processing_delay(0.2, 0.4)

                    # Visualization Section with advanced animation
                    with st.expander("📈 Market Position Visualization",
                                     expanded=True):
                        show_advanced_loading_animation(
                            "chart_brewing", chart_type="Market Position")
                        position_fig = go.Figure()

                        if position_metric == "Price":
                            y_data = centre_df['Sales Price(Kg)']
                            title = 'Price Levels Over Time'
                            y_title = 'Price (₹/Kg)'
                            metric_name = 'Price Levels'
                        elif position_metric == "Volume":
                            y_data = centre_df['Sold Qty (Ton)'] + centre_df[
                                'Unsold Qty (Ton)']
                            title = 'Volume Levels Over Time'
                            y_title = 'Volume (Tons)'
                            metric_name = 'Volume Levels'
                        else:  # Efficiency
                            y_data = centre_df['Sold Qty (Ton)'] / (
                                centre_df['Sold Qty (Ton)'] +
                                centre_df['Unsold Qty (Ton)'])
                            title = 'Market Efficiency Over Time'
                            y_title = 'Efficiency Ratio'
                            metric_name = 'Efficiency Levels'

                        position_fig.add_trace(
                            go.Scatter(x=centre_df['Sale No'],
                                       y=y_data,
                                       name=metric_name,
                                       line=dict(color='#1F4E79', width=2)))

                        # Update layout
                        position_fig.update_layout(title=title,
                                                   xaxis_title='Sale Number',
                                                   yaxis_title=y_title,
                                                   template='plotly_white',
                                                   height=300)

                        # Optimized chart configuration
                        position_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(position_fig,
                                        use_container_width=True,
                                        config=position_config,
                                        key="position_chart")

                    # Analysis Section
                    with st.expander("📊 Position Analysis Insights",
                                     expanded=True):
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Current Level", f"{y_data.iloc[-1]:.2f}",
                                f"{(y_data.iloc[-1] - y_data.iloc[-2]):.2f}")
                        with col2:
                            st.metric("Average", f"{y_data.mean():.2f}")
                        with col3:
                            st.metric("Volatility", f"{y_data.std():.2f}")

                        st.divider()
                        # Display insights
                        levels_insights = analyze_levels(
                            df_selected, selected_centres[0])
                        for insight in levels_insights:
                            st.write(insight)
                else:
                    st.info(
                        "Please select a single market for position analysis")

            # Market Trends Analysis Tab
            with tabs[1]:  # Market Trends
                if len(selected_centres) == 1:
                    # Controls Section - Mobile Optimized
                    with st.expander("📈 Options", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            trend_metric = st.selectbox(
                                "Metric", [
                                    "Sold/Total Ratio",
                                    "Price/Volume Correlation"
                                ],
                                key="trend_metric",
                                help="Choose trend analysis metric")
                        with col2:
                            window_size = st.number_input(
                                "Window",
                                min_value=2,
                                max_value=10,
                                value=3,
                                help="Sales window for calculations",
                                key="correlation_window")

                    # Data Processing
                    centre_df = df_selected[df_selected['Centre'] ==
                                            selected_centres[0]].copy()
                    centre_df = centre_df.sort_values('Sale No')

                    # Price Trend Visualization
                    with st.expander("📊 Price Trend Analysis", expanded=True):
                        price_fig = go.Figure()

                        # Add actual price line
                        price_fig.add_trace(
                            go.Scatter(x=centre_df['Sale No'],
                                       y=centre_df['Sales Price(Kg)'],
                                       name='Actual Price',
                                       line=dict(color='#1F4E79', width=2)))

                        # Add trend line
                        z = np.polyfit(range(len(centre_df)),
                                       centre_df['Sales Price(Kg)'], 1)
                        p = np.poly1d(z)
                        price_fig.add_trace(
                            go.Scatter(x=centre_df['Sale No'],
                                       y=p(range(len(centre_df))),
                                       name='Trend Line',
                                       line=dict(color='#FF9966',
                                                 width=2,
                                                 dash='dash')))

                        # Update layout for price trend
                        price_fig.update_layout(title='Price Trend Analysis',
                                                xaxis_title='Sale No',
                                                yaxis_title='Price (₹/Kg)',
                                                template='plotly_white',
                                                height=300,
                                                showlegend=True,
                                                legend=dict(orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02,
                                                            xanchor="right",
                                                            x=1))

                        # Optimized chart configuration
                        price_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(price_fig,
                                        use_container_width=True,
                                        config=price_config,
                                        key="price_trends_chart")

                    # Market Efficiency Visualization
                    with st.expander("📈 Market Efficiency Analysis",
                                     expanded=True):
                        efficiency_fig = go.Figure()

                        if trend_metric == "Sold/Total Ratio":
                            # Calculate efficiency ratio (Sold/Total)
                            centre_df[
                                'Efficiency'] = centre_df['Sold Qty (Ton)'] / (
                                    centre_df['Sold Qty (Ton)'] +
                                    centre_df['Unsold Qty (Ton)'])

                            efficiency_fig.add_trace(
                                go.Scatter(x=centre_df['Sale No'],
                                           y=centre_df['Efficiency'],
                                           name='Sold/Total Ratio',
                                           line=dict(color='#2E8B57',
                                                     width=2)))

                            title = 'Market Efficiency - Sold/Total Ratio'
                            y_title = 'Ratio'

                        else:  # Price/Volume Correlation
                            # Calculate rolling correlation between price and volume
                            centre_df['Total_Volume'] = centre_df[
                                'Sold Qty (Ton)'] + centre_df[
                                    'Unsold Qty (Ton)']
                            centre_df['Price_Volume_Corr'] = centre_df[
                                'Sales Price(Kg)'].rolling(
                                    window=window_size).corr(
                                        centre_df['Total_Volume'])

                            efficiency_fig.add_trace(
                                go.Scatter(x=centre_df['Sale No'],
                                           y=centre_df['Price_Volume_Corr'],
                                           name='Price/Volume Correlation',
                                           line=dict(color='#2E8B57',
                                                     width=2)))

                            title = 'Market Efficiency - Price/Volume Correlation'
                            y_title = 'Correlation Coefficient'

                        # Update layout for efficiency trend
                        efficiency_fig.update_layout(title=title,
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
                                                         x=1))

                        # Optimized chart configuration
                        efficiency_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(efficiency_fig,
                                        use_container_width=True,
                                        config=efficiency_config,
                                        key="efficiency_chart")

                    # Insights Section
                    with st.expander("📊 Trend Analysis Insights",
                                     expanded=True):
                        # Summary metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            price_change = centre_df['Sales Price(Kg)'].iloc[
                                -1] - centre_df['Sales Price(Kg)'].iloc[0]
                            st.metric(
                                "Overall Price Change", f"₹{price_change:.2f}",
                                f"{(price_change / centre_df['Sales Price(Kg)'].iloc[0] * 100):.1f}%"
                            )
                        with col2:
                            avg_efficiency = centre_df['Sold Qty (Ton)'].sum(
                            ) / (centre_df['Sold Qty (Ton)'].sum() +
                                 centre_df['Unsold Qty (Ton)'].sum())
                            st.metric("Average Market Efficiency",
                                      f"{avg_efficiency:.1%}")

                        st.divider()
                        # Display insights
                        trends_insights = analyze_trends(
                            df_selected, selected_centres[0])
                        for insight in trends_insights:
                            st.write(insight)
                else:
                    st.info(
                        "Please select a single market for trends analysis")

            # Comparative Analysis Tab
            with tabs[2]:  # Comparative Analysis
                if len(selected_centres) == 1:
                    # Controls Section
                    with st.expander("🔄 Comparison Controls", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            comparative_metric = st.selectbox(
                                "Compare By",
                                ["Price", "Volume", "Efficiency"],
                                key="comparative_metric",
                                help="Select the metric for market comparison")
                        with col2:
                            time_window = st.number_input(
                                "Time Window",
                                min_value=1,
                                max_value=12,
                                value=6,
                                help="Number of sales to include in comparison",
                                key="compare_time_window")
                        st.divider()
                        st.info(
                            "✨ Comparing with similar markets for meaningful insights"
                        )

                    # Market Comparison Section
                    with st.expander("📊 Market Comparison", expanded=True):
                        comparative_fig = go.Figure()
                        region, tea_type = selected_centres[0].split(' CTC ')

                        # Get all markets of same type
                        similar_markets = [
                            market
                            for market in df_selected['Centre'].unique()
                            if market.split(' CTC ')[1] == tea_type
                        ]

                        for market in similar_markets:
                            market_df = df_selected[df_selected['Centre'] ==
                                                    market].copy()

                            if comparative_metric == "Price":
                                y_data = market_df['Sales Price(Kg)']
                                title = f'Price Comparison - {tea_type} Markets'
                                y_title = 'Price (₹/Kg)'
                            elif comparative_metric == "Volume":
                                y_data = market_df[
                                    'Sold Qty (Ton)'] + market_df[
                                        'Unsold Qty (Ton)']
                                title = f'Volume Comparison - {tea_type} Markets'
                                y_title = 'Volume (Tons)'
                            else:  # Efficiency
                                y_data = market_df['Sold Qty (Ton)'] / (
                                    market_df['Sold Qty (Ton)'] +
                                    market_df['Unsold Qty (Ton)'])
                                title = f'Efficiency Comparison - {tea_type} Markets'
                                y_title = 'Efficiency Ratio'

                            comparative_fig.add_trace(
                                go.Scatter(x=market_df['Sale No'],
                                           y=y_data,
                                           name=market,
                                           mode='lines'))

                        # Update layout
                        comparative_fig.update_layout(
                            title=title,
                            xaxis_title='Sale Number',
                            yaxis_title=y_title,
                            template='plotly_white',
                            height=300,
                            showlegend=True,
                            legend=dict(orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1))

                        # Optimized chart configuration
                        comparative_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(comparative_fig,
                                        use_container_width=True,
                                        config=comparative_config,
                                        key="comparative_chart")

                    # Correlation Analysis Section
                    with st.expander("🔄 Correlation Analysis", expanded=True):
                        show_advanced_loading_animation("correlation_blending")
                        # Create correlation heatmap
                        correlation_matrix = calculate_correlations(
                            df_selected, selected_centres[0])
                        correlation_fig = go.Figure(
                            data=go.Heatmap(z=correlation_matrix.values,
                                            x=correlation_matrix.columns,
                                            y=correlation_matrix.index,
                                            colorscale='RdBu',
                                            zmid=0))

                        # Update layout
                        correlation_fig.update_layout(
                            title='Metric Correlations',
                            template='plotly_white',
                            height=300)

                        # Optimized chart configuration
                        correlation_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(correlation_fig,
                                        use_container_width=True,
                                        config=correlation_config,
                                        key="correlation_chart")

                    # Insights Section
                    with st.expander("📈 Comparative Analysis Insights",
                                     expanded=True):
                        # Market comparison insights
                        st.subheader("Market Comparison Insights")
                        comparative_insights = analyze_comparatives(
                            df_selected, selected_centres[0])
                        for insight in comparative_insights:
                            st.write(insight)

                        st.divider()

                        # Correlation insights
                        st.subheader("Correlation Insights")
                        correlation_insights = analyze_key_correlations(
                            df_selected, selected_centres[0])
                        for insight in correlation_insights:
                            st.write(insight)
                else:
                    st.info(
                        "Please select a single market for comparative analysis"
                    )

            # Price and Volume Levels Analysis Tab
            with tabs[3]:  # Price and Volume Levels
                if len(selected_centres) == 1:
                    # Controls Section - Enhanced Mobile Layout
                    with st.expander("💰 Analysis Controls", expanded=True):
                        # Date and Distribution Controls
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            date_range = st.select_slider(
                                "Analysis Period",
                                options=list(range(1, 31)),
                                value=(1, 30),
                                key="date_range_slider",
                                help="Select the time period for analysis")

                        with col2:
                            bins = st.number_input(
                                "Price Bins",
                                min_value=5,
                                max_value=50,
                                value=20,
                                key="price_bins_slider",
                                help="Number of groups for price distribution")

                        # Volume Analysis Method
                        st.divider()
                        method_col1, method_col2 = st.columns([3, 1])
                        with method_col1:
                            aggregation_method = st.radio(
                                "Volume Analysis",
                                ["Mean", "Moving Average", "Cumulative"],
                                key="volume_aggregation",
                                horizontal=True,
                                help="Choose how to analyze volume data")
                        with method_col2:
                            if aggregation_method == "Moving Average":
                                window = st.number_input(
                                    "Window",
                                    min_value=2,
                                    max_value=10,
                                    value=3,
                                    key="ma_window",
                                    help="Moving average period")

                    centre_df = df_selected[df_selected['Centre'] ==
                                            selected_centres[0]].copy()

                    # Price Distribution Section
                    with st.expander("📊 Price Distribution Analysis",
                                     expanded=True):
                        # Create price distribution histogram
                        hist_fig = go.Figure()
                        hist_fig.add_trace(
                            go.Histogram(x=centre_df['Sales Price(Kg)'],
                                         nbinsx=bins,
                                         name='Price Distribution',
                                         marker_color='#1F4E79'))

                        hist_fig.update_layout(
                            title='Price Distribution Analysis',
                            xaxis_title='Price (₹/Kg)',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=300)

                        # Optimized chart configuration
                        hist_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(hist_fig,
                                        use_container_width=True,
                                        config=hist_config,
                                        key="histogram_chart")

                        # Add price statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Mean Price",
                                f"₹{centre_df['Sales Price(Kg)'].mean():.2f}")
                        with col2:
                            st.metric(
                                "Median Price",
                                f"₹{centre_df['Sales Price(Kg)'].median():.2f}"
                            )
                        with col3:
                            st.metric(
                                "Price StdDev",
                                f"₹{centre_df['Sales Price(Kg)'].std():.2f}")

                    # Volume Analysis Section
                    with st.expander("📈 Volume Analysis", expanded=True):
                        volume_fig = go.Figure()

                        # Raw volume data
                        total_volume = centre_df['Sold Qty (Ton)'] + centre_df[
                            'Unsold Qty (Ton)']
                        volume_fig.add_trace(
                            go.Scatter(x=centre_df['Sale No'],
                                       y=total_volume,
                                       name='Total Volume',
                                       line=dict(color='#2E8B57', width=2)))

                        # Add aggregation based on selection
                        if aggregation_method == "Moving Average":
                            ma_window = 3  # 3-sale moving average
                            volume_ma = total_volume.rolling(
                                window=ma_window).mean()
                            volume_fig.add_trace(
                                go.Scatter(
                                    x=centre_df['Sale No'],
                                    y=volume_ma,
                                    name=f'{ma_window}-Sale Moving Average',
                                    line=dict(color='#FF9966',
                                              width=2,
                                              dash='dash')))
                        elif aggregation_method == "Cumulative":
                            volume_cumsum = total_volume.cumsum()
                            volume_fig.add_trace(
                                go.Scatter(x=centre_df['Sale No'],
                                           y=volume_cumsum,
                                           name='Cumulative Volume',
                                           line=dict(color='#FF9966',
                                                     width=2,
                                                     dash='dash')))
                        else:  # Mean
                            volume_mean = total_volume.mean()
                            volume_fig.add_trace(
                                go.Scatter(x=centre_df['Sale No'],
                                           y=[volume_mean] * len(centre_df),
                                           name='Mean Volume',
                                           line=dict(color='#FF9966',
                                                     width=2,
                                                     dash='dash')))

                        volume_fig.update_layout(title='Volume Analysis',
                                                 xaxis_title='Sale No',
                                                 yaxis_title='Volume (Tons)',
                                                 template='plotly_white',
                                                 height=300,
                                                 showlegend=True,
                                                 legend=dict(orientation="h",
                                                             yanchor="bottom",
                                                             y=1.02,
                                                             xanchor="right",
                                                             x=1))

                        # Optimized chart configuration
                        volume_config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["drawTools"],
                            "showTips": True,
                            "responsive": True,
                            "staticPlot": False
                        }
                        st.plotly_chart(volume_fig,
                                        use_container_width=True,
                                        config=volume_config,
                                        key="volume_chart")

                        # Add volume statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Volume",
                                      f"{total_volume.sum():.1f} Tons")
                        with col2:
                            st.metric("Average Volume per Sale",
                                      f"{total_volume.mean():.1f} Tons")

                    # Summary Section
                    with st.expander("📋 Summary Insights", expanded=True):
                        st.markdown("""
                            #### Key Findings:
                            
                            **Price Analysis:**
                            - Distribution shape and price ranges
                            - Price volatility and stability periods
                            
                            **Volume Analysis:**
                            - Volume trends and patterns
                            - Peak periods and seasonal variations
                            
                            **Price-Volume Relationship:**
                            - Correlation between price and volume
                            - Market efficiency indicators
                        """)

                else:
                    st.info(
                        "Please select a single market for price and volume analysis"
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Application error: {str(e)}")

except ImportError as e:
    logging.error(f"Import error: {str(e)}")
    st.error(f"Failed to import required libraries: {str(e)}")
except Exception as e:
    logging.error(f"Startup error: {str(e)}")
    st.error(f"Application failed to start: {str(e)}")
