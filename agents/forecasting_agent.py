# agents/forecasting_agent.py
import os
import pandas as pd
from agno.agent import Agent
from agno.tools.file import FileTools

# Import the standalone forecasting function
from forecast_profits import forecast_financial_data

class ForecastingAgent(Agent):
    def __init__(self, model):
        super().__init__(
            name="Forecasting Agent",
            role="Forecast future profits based on historical data",
            model=model,
            tools=[FileTools()],
            instructions=[
                "Analyze historical financial data to predict future profits",
                "Apply forecasting models to find the best fit",
                "Provide forecasts for the next 4 quarters",
                "Document methodology and assumptions"
            ],
            show_tool_calls=True
        )
        
        # Create directory for forecasts
        os.makedirs('output/forecasts', exist_ok=True)
    
    def execute(self, historical_data):
        """
        Execute the forecasting workflow.
        
        Args:
            historical_data (dict): Dictionary containing processed historical datasets
            
        Returns:
            dict: Dictionary containing forecast results for each company
        """
        print("Starting profit forecasting...")
        
        try:
            # Use the standalone forecasting function instead of agent API calls
            forecast_results = forecast_financial_data(historical_data, periods=4)
            print("Forecasting completed successfully!")
            return forecast_results
        except Exception as e:
            print(f"Error during forecasting: {str(e)}")
            # Return an empty dict if forecasting fails
            return {}