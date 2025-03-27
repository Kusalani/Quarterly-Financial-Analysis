import os
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualization_agent')

class VisualizationAgent:
    """
    Agent responsible for creating interactive financial dashboards
    """
    def __init__(self, model=None):
        """Initialize the visualization agent"""
        self.model = model  # Not used but kept for compatibility
        self.data = None
        logger.info("Visualization Agent initialized")
    
    def execute(self, data=None):
        """
        Create interactive dashboard for financial data visualization
        
        Args:
            data (dict): Dictionary of DataFrames with company-specific financial data
            
        Returns:
            dash.Dash: Interactive dashboard application
        """
        logger.info("Creating financial dashboard...")
        
        try:
            # If data is not provided, try to load from files
            if data is None:
                data = self._load_data_from_files()
            
            # Validate and fix the loaded data
            self._validate_data(data)
            
            # Apply scale normalization to make visualizations comparable
            data = self._normalize_scales(data)
            
            # Log data availability
            for company in ["DIPD.N0000", "REXP.N0000", "combined"]:
                if company in data:
                    df = data[company]
                    if df.empty:
                        logger.warning(f"Empty DataFrame for {company}")
                    else:
                        logger.info(f"Data available for {company}: {len(df)} rows, columns: {df.columns.tolist()}")
                        
                        # For combined data, check company distribution
                        if company == "combined" and "Company" in df.columns:
                            dipd_count = (df["Company"] == "DIPD.N0000").sum()
                            rexp_count = (df["Company"] == "REXP.N0000").sum()
                            logger.info(f"Combined data company distribution: DIPD={dipd_count}, REXP={rexp_count}")
                        
                        # Log sample values for important metrics
                        for metric in ['Revenue', 'Net Income']:
                            if metric in df.columns:
                                try:
                                    sample = df[metric].head(2).tolist()
                                    logger.info(f"{company} {metric} sample: {sample}")
                                except Exception as e:
                                    logger.warning(f"Could not get sample for {company} {metric}: {e}")
                else:
                    logger.warning(f"No data for {company}")
            
            # Store the data for use in callbacks
            self.data = data
            
            # Create and configure the dashboard
            dashboard = self._create_dashboard()
            
            logger.info("Dashboard created successfully!")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
    
    def _normalize_scales(self, data):
        """
        Normalize scales between DIPD and REXP data for visualization purposes
        Converting REXP data from millions to show on same scale as DIPD data (billions)
        
        Args:
            data (dict): Dictionary of company DataFrames
            
        Returns:
            dict: Dictionary with normalized data
        """
        logger.info("Checking for scale differences between companies")
        
        if "combined" in data and not data["combined"].empty:
            combined_df = data["combined"].copy()
            
            if "Company" in combined_df.columns:
                # Split data by company
                dipd_data = combined_df[combined_df["Company"] == "DIPD.N0000"]
                rexp_data = combined_df[combined_df["Company"] == "REXP.N0000"]
                
                if not dipd_data.empty and not rexp_data.empty:
                    # Check scale for Revenue as a representative metric
                    if "Revenue" in combined_df.columns:
                        dipd_mean = dipd_data["Revenue"].mean() if not dipd_data["Revenue"].isna().all() else 0
                        rexp_mean = rexp_data["Revenue"].mean() if not rexp_data["Revenue"].isna().all() else 0
                        
                        if dipd_mean > 0 and rexp_mean > 0:
                            ratio = dipd_mean / rexp_mean
                            logger.info(f"Revenue scale ratio (DIPD/REXP): {ratio:.2f}x")
                            
                            # Apply scale normalization if there's a significant difference
                            # Assuming DIPD is in billions and REXP in millions
                            if ratio > 100:  # This indicates a significant scale difference
                                # Since 1 billion = 1000 million, we multiply REXP values by 1000
                                scale_factor = 1000
                                logger.info(f"Applying scale normalization: REXP data will be multiplied by {scale_factor} to match DIPD scale")
                                
                                # Get the mask for REXP data
                                rexp_mask = combined_df["Company"] == "REXP.N0000"
                                
                                # Convert financial metrics for REXP to match DIPD scale
                                for metric in ['Revenue', 'Cost of Goods Sold', 'Gross Profit', 
                                             'Operating Expenses', 'Operating Income', 'Net Income']:
                                    if metric in combined_df.columns:
                                        original_values = combined_df.loc[rexp_mask, metric].copy()
                                        combined_df.loc[rexp_mask, metric] = original_values * scale_factor
                                
                                # Add scale information column
                                if 'scale_note' not in combined_df.columns:
                                    combined_df.loc[rexp_mask, 'scale_note'] = "Values scaled to match DIPD (multiplied by 1000 for comparison)"
                                
                                # Update the dataframe in the dictionary
                                data["combined"] = combined_df
                                
                                # Also update individual REXP data if available
                                if "REXP.N0000" in data and not data["REXP.N0000"].empty:
                                    rexp_df = data["REXP.N0000"].copy()
                                    
                                    for metric in ['Revenue', 'Cost of Goods Sold', 'Gross Profit', 
                                                 'Operating Expenses', 'Operating Income', 'Net Income']:
                                        if metric in rexp_df.columns:
                                            rexp_df[metric] = rexp_df[metric] * scale_factor
                                    
                                    # Add scale note
                                    rexp_df['scale_note'] = "Values scaled to match DIPD (multiplied by 1000 for comparison)"
                                    
                                    # Update the dataframe
                                    data["REXP.N0000"] = rexp_df
                                
                                logger.info("Scale normalization completed")
        
        return data
    
    def _load_data_from_files(self):
        """
        Load data from stored files with improved error handling
        
        Returns:
            dict: Dictionary of company DataFrames
        """
        data = {}
        
        # Try to load from CSV first (more reliable for debugging)
        for company in ["DIPD.N0000", "REXP.N0000", "combined"]:
            csv_path = f'data/processed/{company}_processed.csv'
            pickle_path = f'data/processed/{company}_processed.pkl'
            
            logger.info(f"Attempting to load data for {company}")
            
            # Try CSV first
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # Ensure Date column is properly handled
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    
                    logger.info(f"Successfully loaded {company} from CSV: {len(df)} rows")
                    data[company] = df
                    continue  # Skip to next company if successful
                except Exception as e:
                    logger.error(f"Error loading CSV for {company}: {str(e)}")
                    # Fall through to try pickle
            
            # Try pickle if CSV failed or doesn't exist
            if os.path.exists(pickle_path):
                try:
                    df = pd.read_pickle(pickle_path)
                    # Ensure index is DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        logger.warning(f"Converting {company} index to DatetimeIndex")
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        else:
                            try:
                                df.index = pd.to_datetime(df.index)
                            except Exception as e:
                                logger.warning(f"Failed to convert index to datetime for {company}: {str(e)}")
                    
                    logger.info(f"Successfully loaded {company} from pickle: {len(df)} rows")
                    data[company] = df
                except Exception as e:
                    logger.error(f"Error loading pickle for {company}: {str(e)}")
                    data[company] = pd.DataFrame()
            else:
                logger.warning(f"No data found for {company}")
                data[company] = pd.DataFrame()
            
        return data
    
    def _validate_data(self, data):
        """
        Validate data consistency and fix common issues
        
        Args:
            data (dict): Dictionary of company DataFrames
        """
        for company, df in data.items():
            if df.empty:
                continue
                
            # Check index type
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Index for {company} is not DatetimeIndex, converting now")
                try:
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    else:
                        df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Failed to convert index to datetime for {company}: {str(e)}")
            
            # Ensure numeric columns
            for col in df.columns:
                if col not in ['Company', 'Year', 'Quarter', 'scale_note'] and not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column {col} in {company} is not numeric, converting")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check if Company column exists (for combined data)
            if company == "combined" and "Company" not in df.columns:
                logger.warning("Combined data missing Company column")
                
            # If combined data has both companies, check for significant scale differences
            if company == "combined" and "Company" in df.columns:
                dipd_data = df[df["Company"] == "DIPD.N0000"]
                rexp_data = df[df["Company"] == "REXP.N0000"]
                
                if not dipd_data.empty and not rexp_data.empty:
                    # Check scale for key metrics
                    for metric in ['Revenue', 'Net Income', 'Gross Profit']:
                        if metric in df.columns:
                            dipd_mean = dipd_data[metric].mean() if not dipd_data[metric].isna().all() else 0
                            rexp_mean = rexp_data[metric].mean() if not rexp_data[metric].isna().all() else 0
                            
                            if dipd_mean > 0 and rexp_mean > 0:
                                ratio = dipd_mean / rexp_mean
                                logger.info(f"{metric} scale ratio (DIPD/REXP): {ratio:.2f}x")
                                
                                # Report significant scale differences
                                if ratio > 100 or ratio < 0.01:
                                    logger.warning(f"Significant scale difference in {metric}: {ratio:.2f}x")
    
    def _create_dashboard(self):
        """
        Create the Dash application with interactive components
        
        Returns:
            dash.Dash: Dashboard application
        """
        # Create Dash app with Bootstrap styling
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="CSE Financial Analysis"
        )
        
        # Define color scheme
        colors = {
            "DIPD.N0000": "#1f77b4",  # Blue
            "REXP.N0000": "#ff7f0e",  # Orange
            "background": "#f8f9fa",
            "text": "#212529"
        }
        
        # Create app layout
        app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Colombo Stock Exchange - Quarterly Financial Analysis", 
                           className="text-center my-4"),
                    html.P("Analysis of quarterly financial reports for Dipped Products PLC (DIPD.N0000) and Richard Pieris Exports PLC (REXP.N0000)",
                          className="text-center text-muted")
                ])
            ]),
            
            # Filters
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Company"),
                                    dcc.Dropdown(
                                        id="company-selector",
                                        options=[
                                            {"label": "Dipped Products PLC (DIPD.N0000)", "value": "DIPD.N0000"},
                                            {"label": "Richard Pieris Exports PLC (REXP.N0000)", "value": "REXP.N0000"},
                                            {"label": "Compare Both Companies", "value": "compare"}
                                        ],
                                        value="compare",
                                        clearable=False
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select Date Range"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        min_date_allowed=datetime(2020, 1, 1),
                                        max_date_allowed=datetime(2025, 12, 31),
                                        start_date=datetime(2022, 1, 1),
                                        end_date=datetime(2025, 1, 1),
                                        display_format="YYYY-MM-DD"
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Select Metrics"),
                                    dcc.Dropdown(
                                        id="metrics-selector",
                                        options=[
                                            {"label": "Revenue", "value": "Revenue"},
                                            {"label": "Cost of Goods Sold", "value": "Cost of Goods Sold"},
                                            {"label": "Gross Profit", "value": "Gross Profit"},
                                            {"label": "Operating Income", "value": "Operating Income"},
                                            {"label": "Net Income", "value": "Net Income"},
                                            {"label": "Gross Margin", "value": "Gross Margin"},
                                            {"label": "Operating Margin", "value": "Operating Margin"},
                                            {"label": "Net Margin", "value": "Net Margin"}
                                        ],
                                        value=["Revenue", "Net Income"],
                                        multi=True
                                    )
                                ], width=4)
                            ])
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            # Scale notification
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dbc.Alert(
                            [
                                html.I(className="bi bi-info-circle-fill me-2"),
                                "Note: REXP financial data has been scaled up (multiplied by 1000) for better comparison with DIPD data."
                            ],
                            color="info",
                            className="d-flex align-items-center",
                            id="scale-notification"
                        ),
                        id="scale-notification-container",
                        className="mb-3"
                    )
                ], width=12)
            ]),
            
            # Debug Info (can be hidden in production)
            dbc.Row([
                dbc.Col([
                    html.Div(id="debug-info", className="small text-muted")
                ], width=12)
            ], className="mb-2"),
            
            # Key Financial Metrics Cards
            dbc.Row([
                dbc.Col([
                    html.Div(id="key-metrics-cards")
                ], width=12)
            ], className="mb-4"),
            
            # Main Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quarterly Financial Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="financial-performance-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Margin Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Margin Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="margin-analysis-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Growth Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Growth Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="growth-analysis-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Company Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Company Comparison"),
                        dbc.CardBody([
                            dcc.Graph(id="company-comparison-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("CSE Financial Analysis Dashboard ",
                          className="text-center text-muted")
                ])
            ])
        ], fluid=True, style={"backgroundColor": colors["background"]})
        
        # Define callbacks
        self._define_callbacks(app)
        
        return app
    
    def _define_callbacks(self, app):
        """Define all the callbacks for the dashboard with improved error handling"""
        
        @app.callback(
            Output("scale-notification-container", "className"),
            [Input("company-selector", "value")]
        )
        def toggle_scale_notification(company):
            """Show scale notification when viewing REXP data or comparison"""
            if company == "REXP.N0000" or company == "compare":
                return "mb-3"  # Show the notification 
            else:
                return "d-none mb-3"  # Hide with Bootstrap d-none class
        
        @app.callback(
            Output("debug-info", "children"),
            [Input("company-selector", "value"),
             Input("metrics-selector", "value")]
        )
        def update_debug_info(company, metrics):
            """Debug information to help diagnose issues"""
            debug_info = []
            
            # Show data availability
            if company == "compare":
                # For compare view, show info from combined dataset
                if "combined" in self.data and not self.data["combined"].empty:
                    df = self.data["combined"]
                    
                    if "Company" in df.columns:
                        dipd_count = (df["Company"] == "DIPD.N0000").sum()
                        rexp_count = (df["Company"] == "REXP.N0000").sum()
                        debug_info.append(f"Combined data: {len(df)} total rows")
                        debug_info.append(f"DIPD records: {dipd_count}, REXP records: {rexp_count}")
                        
                        # Show scale information if available
                        if 'scale_note' in df.columns:
                            scale_notes = df.loc[df["Company"] == "REXP.N0000", 'scale_note'].unique()
                            if len(scale_notes) > 0:
                                debug_info.append(f"Scale note: {scale_notes[0]}")
                        
                        # Show date range
                        try:
                            date_range = f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                            debug_info.append(date_range)
                        except Exception as e:
                            debug_info.append("Date range: Unable to determine")
                    else:
                        debug_info.append("Combined data missing Company column")
                else:
                    debug_info.append("No combined data available")
            else:
                # For individual company view
                if company in self.data and not self.data[company].empty:
                    df = self.data[company]
                    debug_info.append(f"{company} data: {len(df)} rows")
                    
                    # Show scale information if available
                    if company == "REXP.N0000" and 'scale_note' in df.columns:
                        scale_notes = df['scale_note'].unique()
                        if len(scale_notes) > 0:
                            debug_info.append(f"Scale note: {scale_notes[0]}")
                    
                    # Show date range
                    try:
                        date_range = f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                        debug_info.append(date_range)
                    except Exception as e:
                        debug_info.append("Date range: Unable to determine")
                else:
                    debug_info.append(f"No data available for {company}")
            
            # Show metrics availability
            debug_info.append("\nMetrics available:")
            for metric in metrics:
                if company == "compare":
                    if "combined" in self.data and metric in self.data["combined"].columns:
                        debug_info.append(f"✓ {metric}")
                    else:
                        debug_info.append(f"✗ {metric}")
                else:
                    if company in self.data and metric in self.data[company].columns:
                        debug_info.append(f"✓ {metric}")
                    else:
                        debug_info.append(f"✗ {metric}")
            
            return html.Pre("\n".join(debug_info))
        
        @app.callback(
            Output("key-metrics-cards", "children"),
            [Input("company-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_key_metrics(company, start_date, end_date):
            """Update the key metrics cards based on selections"""
            try:
                # Convert string dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Filter data based on company selection and date range
                if company == "compare":
                    # Use combined data for comparison
                    if "combined" not in self.data or self.data["combined"].empty:
                        return [dbc.Alert("No combined data available", color="warning")]
                    
                    df = self.data["combined"].copy()
                    
                    # Filter by date range
                    date_mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[date_mask]
                    
                    if filtered_df.empty:
                        return [dbc.Alert("No data available for the selected date range", color="warning")]
                    
                    # Calculate metrics for each company
                    if "Company" in filtered_df.columns:
                        dipd_df = filtered_df[filtered_df["Company"] == "DIPD.N0000"]
                        rexp_df = filtered_df[filtered_df["Company"] == "REXP.N0000"]
                        
                        # Create average metrics across both companies
                        metrics = {}
                        
                        # Function to safely calculate average
                        def safe_mean(df1, df2, column):
                            vals = []
                            if column in df1.columns and not df1[column].isna().all():
                                vals.append(df1[column].mean())
                            if column in df2.columns and not df2[column].isna().all():
                                vals.append(df2[column].mean())
                            return sum(vals) / len(vals) if vals else None
                        
                        # Calculate averages
                        metrics["Revenue"] = safe_mean(dipd_df, rexp_df, "Revenue")
                        metrics["Net Income"] = safe_mean(dipd_df, rexp_df, "Net Income")
                        metrics["Gross Margin"] = safe_mean(dipd_df, rexp_df, "Gross Margin")
                        metrics["Net Margin"] = safe_mean(dipd_df, rexp_df, "Net Margin")
                    else:
                        # Fall back to direct means if Company column is missing
                        metrics = {
                            "Revenue": filtered_df["Revenue"].mean() if "Revenue" in filtered_df.columns else None,
                            "Net Income": filtered_df["Net Income"].mean() if "Net Income" in filtered_df.columns else None,
                            "Gross Margin": filtered_df["Gross Margin"].mean() if "Gross Margin" in filtered_df.columns else None,
                            "Net Margin": filtered_df["Net Margin"].mean() if "Net Margin" in filtered_df.columns else None
                        }
                else:
                    # Use individual company data
                    if company not in self.data or self.data[company].empty:
                        return [dbc.Alert(f"No data available for {company}", color="warning")]
                    
                    df = self.data[company].copy()
                    
                    # Filter by date range
                    date_mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[date_mask]
                    
                    if filtered_df.empty:
                        return [dbc.Alert(f"No data available for {company} in the selected date range", color="warning")]
                    
                    # Calculate metrics
                    metrics = {
                        "Revenue": filtered_df["Revenue"].mean() if "Revenue" in filtered_df.columns else None,
                        "Net Income": filtered_df["Net Income"].mean() if "Net Income" in filtered_df.columns else None,
                        "Gross Margin": filtered_df["Gross Margin"].mean() if "Gross Margin" in filtered_df.columns else None,
                        "Net Margin": filtered_df["Net Margin"].mean() if "Net Margin" in filtered_df.columns else None
                    }
                
                # Create cards
                cards = []
                
                # Revenue card
                if metrics["Revenue"] is not None:
                    revenue_card = dbc.Card([
                        dbc.CardBody([
                            html.H5("Average Revenue", className="card-title"),
                            html.H3(f"{metrics['Revenue']:,.2f}", className="text-primary"),
                            html.P("Average quarterly revenue over the selected period", className="card-text")
                        ])
                    ], className="text-center")
                    cards.append(dbc.Col(revenue_card, width=3))
                
                # Net Income card
                if metrics["Net Income"] is not None:
                    net_income_card = dbc.Card([
                        dbc.CardBody([
                            html.H5("Average Net Income", className="card-title"),
                            html.H3(f"{metrics['Net Income']:,.2f}", className="text-success"),
                            html.P("Average quarterly net income over the selected period", className="card-text")
                        ])
                    ], className="text-center")
                    cards.append(dbc.Col(net_income_card, width=3))
                
                # Gross Margin card
                if metrics["Gross Margin"] is not None:
                    gross_margin_card = dbc.Card([
                        dbc.CardBody([
                            html.H5("Average Gross Margin", className="card-title"),
                            html.H3(f"{metrics['Gross Margin']:.2f}%", className="text-info"),
                            html.P("Average gross margin percentage over the selected period", className="card-text")
                        ])
                    ], className="text-center")
                    cards.append(dbc.Col(gross_margin_card, width=3))
                
                # Net Margin card
                if metrics["Net Margin"] is not None:
                    net_margin_card = dbc.Card([
                        dbc.CardBody([
                            html.H5("Average Net Margin", className="card-title"),
                            html.H3(f"{metrics['Net Margin']:.2f}%", className="text-warning"),
                            html.P("Average net margin percentage over the selected period", className="card-text")
                        ])
                    ], className="text-center")
                    cards.append(dbc.Col(net_margin_card, width=3))
                
                return dbc.Row(cards) if cards else dbc.Alert("No metrics available", color="warning")
                
            except Exception as e:
                logger.error(f"Error updating key metrics: {str(e)}")
                return dbc.Alert(f"Error calculating metrics: {str(e)}", color="danger")
        
        @app.callback(
            Output("financial-performance-chart", "figure"),
            [Input("company-selector", "value"),
             Input("metrics-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_financial_performance(company, metrics, start_date, end_date):
            """Update the financial performance chart based on selections"""
            try:
                # Convert string dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Determine which data to use
                # Determine which data to use
                if company == "compare":
                    # Use combined data with filtering by company
                    if "combined" not in self.data or self.data["combined"].empty:
                        return go.Figure().update_layout(
                            title="No combined data available",
                            template="plotly_white"
                        )
                    
                    combined_df = self.data["combined"].copy()
                    date_mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
                    filtered_df = combined_df[date_mask]
                    
                    if "Company" not in filtered_df.columns:
                        return go.Figure().update_layout(
                            title="Combined data missing Company column",
                            template="plotly_white"
                        )
                    
                    # Split by company
                    dipd_df = filtered_df[filtered_df["Company"] == "DIPD.N0000"]
                    rexp_df = filtered_df[filtered_df["Company"] == "REXP.N0000"]
                    
                    company_dfs = {
                        "DIPD.N0000": dipd_df,
                        "REXP.N0000": rexp_df
                    }
                    
                    companies = ["DIPD.N0000", "REXP.N0000"]
                else:
                    # Use individual company data
                    if company not in self.data or self.data[company].empty:
                        return go.Figure().update_layout(
                            title=f"No data available for {company}",
                            template="plotly_white"
                        )
                    
                    df = self.data[company].copy()
                    date_mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[date_mask]
                    
                    company_dfs = {company: filtered_df}
                    companies = [company]
                
                # Check if we have any data to plot
                if all(df.empty for df in company_dfs.values()):
                    return go.Figure().update_layout(
                        title="No data available for the selected criteria",
                        template="plotly_white"
                    )
                
                # Configure secondary axis usage
                use_secondary_axis = len(metrics) > 1
                secondary_metrics = metrics[1:] if use_secondary_axis else []
                
                # Colors for each company
                colors = {
                    "DIPD.N0000": "#1f77b4",  # Blue
                    "REXP.N0000": "#ff7f0e"   # Orange
                }
                
                # Track if any traces were added
                traces_added = False
                
                # Add traces for each company and metric
                for comp in companies:
                    if comp not in company_dfs or company_dfs[comp].empty:
                        continue
                    
                    df = company_dfs[comp]
                    
                    for i, metric in enumerate(metrics):
                        if metric not in df.columns:
                            continue
                        
                        # Skip if all values are NaN
                        if df[metric].isna().all():
                            continue
                        
                        # Determine if this metric should use secondary y-axis
                        use_secondary = (metric in secondary_metrics)
                        
                        # Add trace
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[metric],
                                name=f"{comp} - {metric}",
                                mode="lines+markers",
                                line=dict(color=colors[comp], width=2, dash="solid" if i == 0 else "dash"),
                                marker=dict(size=8)
                            ),
                            secondary_y=use_secondary
                        )
                        
                        traces_added = True
                
                # If no traces were added, return an empty figure
                if not traces_added:
                    return go.Figure().update_layout(
                        title="No data available for the selected metrics",
                        template="plotly_white"
                    )
                
                # Update layout
                primary_metric = metrics[0] if metrics else ""
                secondary_metric = metrics[1] if len(metrics) > 1 else ""
                
                title_text = "Quarterly Financial Performance"
                if company == "compare" or company == "REXP.N0000":
                    title_text += " (REXP data scaled up by 1000x for comparison)"
                
                fig.update_layout(
                    title=title_text,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                fig.update_xaxes(title_text="Quarter", tickangle=45)
                fig.update_yaxes(title_text=primary_metric, secondary_y=False)
                
                if use_secondary_axis:
                    fig.update_yaxes(title_text=secondary_metric, secondary_y=True)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating financial performance chart: {str(e)}")
                return go.Figure().update_layout(
                    title=f"Error: {str(e)}",
                    template="plotly_white"
                )
        
        @app.callback(
            Output("margin-analysis-chart", "figure"),
            [Input("company-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_margin_analysis(company, start_date, end_date):
            """Update the margin analysis chart based on selections"""
            try:
                # Convert string dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Create figure
                fig = go.Figure()
                
                # Determine which data to use
                if company == "compare":
                    # Use combined data with filtering by company
                    if "combined" not in self.data or self.data["combined"].empty:
                        return go.Figure().update_layout(
                            title="No combined data available",
                            template="plotly_white"
                        )
                    
                    combined_df = self.data["combined"].copy()
                    date_mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
                    filtered_df = combined_df[date_mask]
                    
                    if "Company" not in filtered_df.columns:
                        return go.Figure().update_layout(
                            title="Combined data missing Company column",
                            template="plotly_white"
                        )
                    
                    # Split by company
                    dipd_df = filtered_df[filtered_df["Company"] == "DIPD.N0000"]
                    rexp_df = filtered_df[filtered_df["Company"] == "REXP.N0000"]
                    
                    company_dfs = {
                        "DIPD.N0000": dipd_df,
                        "REXP.N0000": rexp_df
                    }
                    
                    companies = ["DIPD.N0000", "REXP.N0000"]
                else:
                    # Use individual company data
                    if company not in self.data or self.data[company].empty:
                        return go.Figure().update_layout(
                            title=f"No data available for {company}",
                            template="plotly_white"
                        )
                    
                    df = self.data[company].copy()
                    date_mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[date_mask]
                    
                    company_dfs = {company: filtered_df}
                    companies = [company]
                
                # Check if we have any data to plot
                if all(df.empty for df in company_dfs.values()):
                    return go.Figure().update_layout(
                        title="No data available for the selected criteria",
                        template="plotly_white"
                    )
                
                # Margin metrics to plot
                margin_metrics = ["Gross Margin", "Operating Margin", "Net Margin"]
                
                # Colors for each company
                colors = {
                    "DIPD.N0000": "#1f77b4",  # Blue
                    "REXP.N0000": "#ff7f0e"   # Orange
                }
                
                # Track if any traces were added
                traces_added = False
                
                # Add traces for each company and margin metric
                for comp in companies:
                    if comp not in company_dfs or company_dfs[comp].empty:
                        continue
                    
                    df = company_dfs[comp]
                    
                    for metric in margin_metrics:
                        if metric not in df.columns:
                            continue
                        
                        # Skip if all values are NaN
                        if df[metric].isna().all():
                            continue
                        
                        # Add trace
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[metric],
                                name=f"{comp} - {metric}",
                                mode="lines+markers",
                                line=dict(color=colors[comp], width=2),
                                marker=dict(size=8)
                            )
                        )
                        
                        traces_added = True
                
                # If no traces were added, return an empty figure
                if not traces_added:
                    return go.Figure().update_layout(
                        title="No margin data available",
                        template="plotly_white"
                    )
                
                # Update layout
                fig.update_layout(
                    title="Margin Analysis",
                    xaxis_title="Quarter",
                    yaxis_title="Margin (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating margin analysis chart: {str(e)}")
                return go.Figure().update_layout(
                    title=f"Error: {str(e)}",
                    template="plotly_white"
                )
        
        @app.callback(
            Output("growth-analysis-chart", "figure"),
            [Input("company-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_growth_analysis(company, start_date, end_date):
            """Update the growth analysis chart based on selections"""
            try:
                # Convert string dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Create figure
                fig = go.Figure()
                
                # Determine which data to use
                if company == "compare":
                    # Use combined data with filtering by company
                    if "combined" not in self.data or self.data["combined"].empty:
                        return go.Figure().update_layout(
                            title="No combined data available",
                            template="plotly_white"
                        )
                    
                    combined_df = self.data["combined"].copy()
                    date_mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
                    filtered_df = combined_df[date_mask]
                    
                    if "Company" not in filtered_df.columns:
                        return go.Figure().update_layout(
                            title="Combined data missing Company column",
                            template="plotly_white"
                        )
                    
                    # Split by company
                    dipd_df = filtered_df[filtered_df["Company"] == "DIPD.N0000"]
                    rexp_df = filtered_df[filtered_df["Company"] == "REXP.N0000"]
                    
                    company_dfs = {
                        "DIPD.N0000": dipd_df,
                        "REXP.N0000": rexp_df
                    }
                    
                    companies = ["DIPD.N0000", "REXP.N0000"]
                else:
                    # Use individual company data
                    if company not in self.data or self.data[company].empty:
                        return go.Figure().update_layout(
                            title=f"No data available for {company}",
                            template="plotly_white"
                        )
                    
                    df = self.data[company].copy()
                    date_mask = (df.index >= start_date) & (df.index <= end_date)
                    filtered_df = df[date_mask]
                    
                    company_dfs = {company: filtered_df}
                    companies = [company]
                
                # Check if we have any data to plot
                if all(df.empty for df in company_dfs.values()):
                    return go.Figure().update_layout(
                        title="No data available for the selected criteria",
                        template="plotly_white"
                    )
                
                # Growth metrics to plot
                growth_metrics = ["Revenue_QoQ_Growth", "Net_Income_QoQ_Growth"]
                
                # Colors for each company
                colors = {
                    "DIPD.N0000": "#1f77b4",  # Blue
                    "REXP.N0000": "#ff7f0e"   # Orange
                }
                
                # Track if any traces were added
                traces_added = False
                
                # Add traces for each company and growth metric
                for comp in companies:
                    if comp not in company_dfs or company_dfs[comp].empty:
                        continue
                    
                    df = company_dfs[comp]
                    
                    for metric in growth_metrics:
                        # Calculate growth metrics if they don't exist
                        if metric not in df.columns:
                            if metric == "Revenue_QoQ_Growth" and "Revenue" in df.columns:
                                df[metric] = df["Revenue"].pct_change() * 100
                            elif metric == "Net_Income_QoQ_Growth" and "Net Income" in df.columns:
                                df[metric] = df["Net Income"].pct_change() * 100
                        
                        if metric not in df.columns or df[metric].isna().all():
                            continue
                        
                        # Get display name
                        display_name = "Revenue QoQ Growth" if metric == "Revenue_QoQ_Growth" else "Net Income QoQ Growth"
                        
                        # Add trace
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df[metric],
                                name=f"{comp} - {display_name}",
                                marker_color=colors[comp],
                                opacity=0.7 if metric == "Net_Income_QoQ_Growth" else 1.0
                            )
                        )
                        
                        traces_added = True
                
                # If no traces were added, return an empty figure
                if not traces_added:
                    return go.Figure().update_layout(
                        title="No growth data available",
                        template="plotly_white"
                    )
                
                # Update layout
                fig.update_layout(
                    title="Quarter-over-Quarter Growth Analysis",
                    xaxis_title="Quarter",
                    yaxis_title="Growth Rate (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    barmode="group",
                    template="plotly_white",
                    height=500
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating growth analysis chart: {str(e)}")
                return go.Figure().update_layout(
                    title=f"Error: {str(e)}",
                    template="plotly_white"
                )
        
        @app.callback(
            Output("company-comparison-chart", "figure"),
            [Input("metrics-selector", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_company_comparison(metrics, start_date, end_date):
            """Update the company comparison chart based on selections"""
            try:
                # Convert string dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Check if we have combined data
                if "combined" not in self.data or self.data["combined"].empty:
                    return go.Figure().update_layout(
                        title="No combined data available for comparison",
                        template="plotly_white"
                    )
                
                # Filter combined data by date range
                combined_df = self.data["combined"].copy()
                date_mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
                filtered_df = combined_df[date_mask]
                
                # Check if we have the Company column
                if "Company" not in filtered_df.columns:
                    return go.Figure().update_layout(
                        title="Combined data missing Company column",
                        template="plotly_white"
                    )
                
                # Split by company
                dipd_df = filtered_df[filtered_df["Company"] == "DIPD.N0000"]
                rexp_df = filtered_df[filtered_df["Company"] == "REXP.N0000"]
                
                # Check if we have data for both companies
                if dipd_df.empty or rexp_df.empty:
                    return go.Figure().update_layout(
                        title="Missing data for one or both companies in the selected date range",
                        template="plotly_white"
                    )
                
                # Create subplots for each metric
                subplot_titles = [f"{metric} Comparison" for metric in metrics]
                fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True,
                                    subplot_titles=subplot_titles)
                
                # Track if any traces were added
                traces_added = False
                
                # Plot each metric as a separate subplot
                for i, metric in enumerate(metrics):
                    row = i + 1
                    
                    # Check if metric is available in both datasets
                    if metric in dipd_df.columns and metric in rexp_df.columns:
                        # Check if there's any data for this metric
                        if not dipd_df[metric].isna().all() and not rexp_df[metric].isna().all():
                            # Add DIPD data
                            fig.add_trace(
                                go.Scatter(
                                    x=dipd_df.index,
                                    y=dipd_df[metric],
                                    name=f"DIPD.N0000 - {metric}",
                                    mode="lines+markers",
                                    line=dict(color="#1f77b4", width=2),
                                    marker=dict(size=8),
                                    showlegend=(row == 1)  # Only show legend for first row
                                ),
                                row=row, col=1
                            )
                            
                            # Add REXP data
                            fig.add_trace(
                                go.Scatter(
                                    x=rexp_df.index,
                                    y=rexp_df[metric],
                                    name=f"REXP.N0000 - {metric}",
                                    mode="lines+markers",
                                    line=dict(color="#ff7f0e", width=2),
                                    marker=dict(size=8),
                                    showlegend=(row == 1)  # Only show legend for first row
                                ),
                                row=row, col=1
                            )
                            
                            traces_added = True
                
                # If no traces were added, return an empty figure
                if not traces_added:
                    return go.Figure().update_layout(
                        title="No data available for comparison",
                        template="plotly_white"
                    )
                
                # Update layout
                fig.update_layout(
                    title="Company Comparison by Metric (REXP data scaled up by 1000x for comparison)",
                    height=300*len(metrics),  # Adjust height based on number of metrics
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                # Update y-axis titles
                for i, metric in enumerate(metrics):
                    fig.update_yaxes(title_text=metric, row=i+1, col=1)
                
                # Update x-axis title for bottom subplot only
                fig.update_xaxes(title_text="Quarter", row=len(metrics), col=1)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating company comparison chart: {str(e)}")
                return go.Figure().update_layout(
                    title=f"Error: {str(e)}",
                    template="plotly_white"
                )
                    