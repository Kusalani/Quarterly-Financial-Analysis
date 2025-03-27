# agents/forecasting_agent.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.file import FileTools

# Load environment variables
load_dotenv()

class ForecastingAgent:
    def __init__(self, model=None):
        # Initialize model based on available APIs
        if model is None:
            if os.getenv("GROQ_API_KEY"):
                model = Groq(id="llama-3-8b-8192")
                print("Using Groq model for forecasting")
            elif os.getenv("GEMINI_API_KEY"):
                model = Gemini()
                print("Using Gemini model for forecasting")
            else:
                print("Warning: No model API keys found. Using fallback forecasting.")
        
        # Initialize the agent
        self.agent = Agent(
            model=model,
            role="Financial forecasting expert",
            instructions=[
                "Analyze financial data and generate accurate forecasts",
                "Provide detailed insights on market trends",
                "Identify growth opportunities and potential risks"
            ],
            tools=[FileTools()],  # Only using FileTools which is available
            markdown=True
        )
        
        # Create output directories
        os.makedirs('output/forecasts/plots', exist_ok=True)
        os.makedirs('output/reports', exist_ok=True)
    
    def execute(self, historical_data):
        """
        Generate forecasts for financial data
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical financial data
            
        Returns:
            dict: Forecast results
        """
        print("Starting agent-based forecasting...")
        
        # Process companies
        forecast_results = {}
        companies = [key for key in historical_data.keys() if key != 'combined']
        
        # Create a detailed prompt with the financial data
        prompt = self._create_forecast_prompt(historical_data, companies)
        
        # Get the forecast from the agent
        analysis = self.agent.response(prompt)
        
        # Process the agent's response to extract forecasts
        forecast_results = self._process_forecast_response(analysis, historical_data)
        
        # Generate visualizations
        self._create_visualizations(historical_data, forecast_results)
        
        # Save the report
        self._save_report(analysis)
        
        print("Forecasting completed successfully!")
        return forecast_results
    
    def _create_forecast_prompt(self, historical_data, companies):
        """Create a detailed prompt with the financial data"""
        prompt = """You are a financial forecasting expert. Analyze the following financial data and provide:
1. A detailed forecast for the next 4 quarters for each company
2. Strategic insights and investment recommendations
3. Comparative analysis between companies

Here is the financial data:
"""
        
        # Add data for each company
        for company in companies:
            data = historical_data[company]
            
            prompt += f"\n## {company}\n"
            prompt += f"Data points: {len(data)} quarters\n"
            prompt += f"Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n\n"
            
            # Add recent quarterly data
            prompt += "Recent quarterly data:\n"
            recent_data = data.tail(4).reset_index()
            for _, row in recent_data.iterrows():
                prompt += f"- {row['Date'].strftime('%Y-%m-%d')}: "
                prompt += f"Revenue: {row.get('Revenue', 'N/A')}, "
                prompt += f"Net Income: {row.get('Net Income', 'N/A')}, "
                prompt += f"Gross Profit: {row.get('Gross Profit', 'N/A')}, "
                prompt += f"Operating Income: {row.get('Operating Income', 'N/A')}\n"
            
            # Add key metrics
            prompt += "\nKey metrics (average):\n"
            for metric in ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']:
                if metric in data.columns:
                    prompt += f"- {metric}: {data[metric].mean():,.2f}\n"
            
            # Add growth metrics
            prompt += "\nGrowth rates:\n"
            for metric in ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']:
                if metric in data.columns:
                    pct_changes = data[metric].pct_change().dropna() * 100
                    avg_growth = pct_changes.mean()
                    prompt += f"- {metric} average growth rate: {avg_growth:.2f}%\n"
        
        # Add instructions for the forecast format
        prompt += """
Please provide your forecast in the following format for each company:

## [Company Name] Forecast

| Metric | Q1 | Q2 | Q3 | Q4 | Confidence |
|--------|---|---|---|---|------------|
| Revenue | value | value | value | value | High/Medium/Low |
| Net Income | value | value | value | value | High/Medium/Low |
| Gross Profit | value | value | value | value | High/Medium/Low |
| Operating Income | value | value | value | value | High/Medium/Low |

Include a brief explanation of your methodology and key insights for each company.
Also provide a comparative analysis and investment recommendations.
"""
        
        return prompt
    
    def _process_forecast_response(self, response, historical_data):
        """
        Extract forecast data from the agent's response
        
        Args:
            response (str): Agent's response text
            historical_data (dict): Original historical data
            
        Returns:
            dict: Structured forecast results
        """
        import re
        
        forecast_results = {}
        companies = [key for key in historical_data.keys() if key != 'combined']
        
        for company in companies:
            forecast_results[company] = {}
            
            # Extract table data for this company
            company_pattern = rf"## {company}[\s\S]*?\n\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|"
            table_match = re.search(company_pattern, response, re.IGNORECASE)
            
            if not table_match:
                # Try alternative pattern
                company_pattern = rf"{company}[\s\S]*?\n\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|"
                table_match = re.search(company_pattern, response, re.IGNORECASE)
            
            if table_match:
                # Found the table header, now extract each row
                table_text = table_match.group(0)
                
                # Extract each metric row
                for metric in ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']:
                    metric_pattern = rf"\|\s*{metric}\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)\s*\|\s*([a-zA-Z]+)\s*\|"
                    metric_match = re.search(metric_pattern, table_text, re.IGNORECASE)
                    
                    if metric_match:
                        # Extract forecasted values
                        q1 = float(metric_match.group(1).replace(',', ''))
                        q2 = float(metric_match.group(2).replace(',', ''))
                        q3 = float(metric_match.group(3).replace(',', ''))
                        q4 = float(metric_match.group(4).replace(',', ''))
                        confidence = metric_match.group(5).lower()
                        
                        # Get last date from historical data
                        last_date = historical_data[company].index[-1]
                        
                        # Create future dates
                        future_dates = [
                            last_date + pd.DateOffset(months=3),
                            last_date + pd.DateOffset(months=6),
                            last_date + pd.DateOffset(months=9),
                            last_date + pd.DateOffset(months=12)
                        ]
                        
                        # Store forecast data
                        forecast_results[company][metric] = {
                            'forecast': pd.Series([q1, q2, q3, q4], index=future_dates),
                            'lower_bound': pd.Series([q1 * 0.9, q2 * 0.9, q3 * 0.9, q4 * 0.9], index=future_dates),
                            'upper_bound': pd.Series([q1 * 1.1, q2 * 1.1, q3 * 1.1, q4 * 1.1], index=future_dates),
                            'confidence': confidence,
                            'methodology': "AI-based forecasting"
                        }
            else:
                # If we couldn't find a table, create fallback forecasts
                for metric in ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']:
                    if metric in historical_data[company].columns:
                        historical = historical_data[company][metric].dropna()
                        
                        if len(historical) > 0:
                            # Calculate average growth rate
                            pct_changes = historical.pct_change().dropna()
                            avg_growth_rate = pct_changes.mean() if not pct_changes.empty else 0.05
                            
                            # Get last date and value
                            last_date = historical_data[company].index[-1]
                            last_value = historical.iloc[-1]
                            
                            # Generate forecast values
                            forecast_values = []
                            current_value = last_value
                            
                            for _ in range(4):
                                current_value = current_value * (1 + avg_growth_rate)
                                forecast_values.append(current_value)
                            
                            # Create future dates
                            future_dates = [
                                last_date + pd.DateOffset(months=3),
                                last_date + pd.DateOffset(months=6),
                                last_date + pd.DateOffset(months=9),
                                last_date + pd.DateOffset(months=12)
                            ]
                            
                            # Store forecast data
                            forecast_results[company][metric] = {
                                'forecast': pd.Series(forecast_values, index=future_dates),
                                'lower_bound': pd.Series([v * 0.85 for v in forecast_values], index=future_dates),
                                'upper_bound': pd.Series([v * 1.15 for v in forecast_values], index=future_dates),
                                'confidence': 'low',
                                'methodology': 'Simple growth extrapolation (fallback)'
                            }
        
        return forecast_results
    
    def _create_visualizations(self, historical_data, forecast_results):
        """Create visualizations for the forecasts"""
        print("Generating forecast visualizations...")
        
        # Get companies
        companies = [key for key in forecast_results.keys() if key != 'combined']
        
        for company in companies:
            for metric in forecast_results[company]:
                # Get historical data
                if metric in historical_data[company].columns:
                    historical = historical_data[company][metric]
                    
                    # Get forecast data
                    forecast = forecast_results[company][metric]['forecast']
                    lower_bound = forecast_results[company][metric]['lower_bound']
                    upper_bound = forecast_results[company][metric]['upper_bound']
                    confidence = forecast_results[company][metric]['confidence']
                    
                    # Create visualization
                    plt.figure(figsize=(12, 6))
                    
                    # Plot historical data
                    plt.plot(historical.index, historical, 'b-', linewidth=2, label='Historical')
                    
                    # Plot forecast
                    plt.plot(forecast.index, forecast, 'r--', linewidth=2, label='Forecast')
                    
                    # Plot confidence interval
                    plt.fill_between(
                        forecast.index,
                        lower_bound,
                        upper_bound,
                        color='red',
                        alpha=0.2,
                        label='Confidence Interval'
                    )
                    
                    # Add confidence level
                    confidence_color = {
                        'high': 'green',
                        'medium': 'orange',
                        'low': 'red'
                    }.get(confidence.lower(), 'gray')
                    
                    plt.figtext(
                        0.02, 0.02,
                        f"Confidence: {confidence.upper()}",
                        color=confidence_color,
                        fontweight='bold',
                        fontsize=10
                    )
                    
                    # Calculate growth
                    latest_historical = historical.iloc[-1]
                    latest_forecast = forecast.iloc[-1]
                    growth = ((latest_forecast / latest_historical) - 1) * 100
                    
                    # Add growth annotation
                    plt.annotate(
                        f"Projected Growth: {growth:.1f}%",
                        xy=(forecast.index[-1], forecast.iloc[-1]),
                        xytext=(15, 15),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
                    )
                    
                    # Add labels and title
                    plt.title(f'{company} - {metric} Forecast', fontsize=14, fontweight='bold')
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel(metric, fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Format x-axis dates
                    plt.gcf().autofmt_xdate()
                    
                    # Save the plot
                    filename = f"{company.replace('.', '_')}_{metric.replace(' ', '_')}_forecast.png"
                    plt.savefig(f'output/forecasts/plots/{filename}', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Create comparative visualizations if multiple companies
        if len(companies) > 1:
            for metric in ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']:
                # Check if we have data for this metric across companies
                companies_with_metric = [c for c in companies if metric in forecast_results[c]]
                
                if len(companies_with_metric) > 1:
                    # Create both normalized and absolute comparisons
                    for plot_type in ['normalized', 'absolute']:
                        plt.figure(figsize=(12, 6))
                        
                        for company in companies_with_metric:
                            # Get historical and forecast data
                            if metric in historical_data[company].columns:
                                historical = historical_data[company][metric]
                                forecast = forecast_results[company][metric]['forecast']
                                
                                if plot_type == 'normalized':
                                    # Normalize to 100 at the first data point
                                    first_value = historical.iloc[0]
                                    hist_values = historical / first_value * 100
                                    fore_values = forecast / first_value * 100
                                    ylabel = f'Normalized {metric} (Base=100)'
                                else:
                                    # Use absolute values
                                    hist_values = historical
                                    fore_values = forecast
                                    ylabel = metric
                                
                                # Plot the data
                                plt.plot(historical.index, hist_values, '-', linewidth=2, label=f'{company} Historical')
                                plt.plot(forecast.index, fore_values, '--', linewidth=2, label=f'{company} Forecast')
                        
                        # Add labels and title
                        plt.title(f'Comparative {metric} Forecast ({plot_type.title()})', fontsize=14, fontweight='bold')
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel(ylabel, fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        
                        # Format x-axis dates
                        plt.gcf().autofmt_xdate()
                        
                        # Save the plot
                        filename = f"comparative_{metric.replace(' ', '_')}_{plot_type}.png"
                        plt.savefig(f'output/forecasts/plots/{filename}', dpi=300, bbox_inches='tight')
                        plt.close()
    
    def _save_report(self, analysis):
        """Save the forecast report"""
        print("Saving forecast report...")
        
        report_path = 'output/reports/profit_forecast_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Profit Forecasting: Methodology, Results, and Strategic Insights\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(analysis)
            
            # Add visualization references
            f.write("\n\n## Forecast Visualizations\n\n")
            f.write("See the 'output/forecasts/plots' directory for all forecast visualizations.\n")
        
        return report_path