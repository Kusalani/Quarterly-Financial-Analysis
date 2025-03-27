#from agents.scraper_agent import ScraperAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.visualization_agent import VisualizationAgent
from agents.forecasting_agent import ForecastingAgent

# Export all agent classes
__all__ = [
    'DataProcessingAgent', 
    'VisualizationAgent', 
    'ForecastingAgent'
]