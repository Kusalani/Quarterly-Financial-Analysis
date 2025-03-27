import os
import sys
import pandas as pd
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cse_financial_analysis')

# Import custom modules
from rexp_financial_scraper import extract_rexp_financial_data, rexp_urls
from dipd_financial_scraper import extract_dipd_financial_data, dipd_urls, convert_to_dataframe
from agents.data_processing_agent import DataProcessingAgent
from agents.visualization_agent import VisualizationAgent
from agents.forecasting_agent import ForecastingAgent

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CSE Financial Analysis Tool')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip the data scraping step')
    parser.add_argument('--skip-processing', action='store_true', help='Skip the data processing step')
    parser.add_argument('--skip-forecasting', action='store_true', help='Skip the profit forecasting step')
    parser.add_argument('--dashboard-only', action='store_true', help='Run only the dashboard')
    parser.add_argument('--no-dashboard', action='store_true', help='Do not run the dashboard')
    return parser.parse_args()

def main():
    """Main execution function"""
    print("=" * 80)
    print("Colombo Stock Exchange (CSE) - Quarterly Financial Analysis")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create required directories
    os.makedirs('data/extracted', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('output/visualizations', exist_ok=True)
    os.makedirs('output/forecasts', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    
    # Dictionary to store financial data
    financial_data = {}
    
    # Data Scraping
    if args.dashboard_only and os.path.exists('data/processed/combined_data.pkl'):
        # If dashboard-only mode and processed data exists, load it
        financial_data = {
            "DIPD.N0000": pd.read_pickle('data/processed/DIPD.N0000_processed.pkl'),
            "REXP.N0000": pd.read_pickle('data/processed/REXP.N0000_processed.pkl'),
            "combined": pd.read_pickle('data/processed/combined_data.pkl')
        }
        logger.info("Loaded pre-processed data")
    elif not args.skip_scraping:
        print("\nStep 1: Extracting financial data from CSE quarterly reports...")
        try:
            # Extract DIPD data 
            dipd_data = extract_dipd_financial_data(dipd_urls)
            dipd_df = convert_to_dataframe("DIPD.N0000", dipd_data)
            
            # Save DIPD DataFrame
            dipd_df.to_csv('data/processed/DIPD.N0000_processed.csv')
            dipd_df.to_pickle('data/processed/DIPD.N0000_processed.pkl')
            
            # Extract REXP data 
            rexp_data = extract_rexp_financial_data(rexp_urls)
            rexp_df = rexp_data.get("REXP.N0000", pd.DataFrame())
            
            # Create combined dataset
            combined_df = pd.concat([dipd_df, rexp_df])
            combined_df.to_csv('data/processed/combined_data.csv')
            combined_df.to_pickle('data/processed/combined_data.pkl')
            
            # Store the data in a dictionary for further processing
            financial_data = {
                "DIPD.N0000": dipd_df,
                "REXP.N0000": rexp_df,
                "combined": combined_df
            }
            
            logger.info("Data extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during data extraction: {str(e)}")
            return
        
        
    # Data Processing
    if not args.skip_processing and not args.dashboard_only:
        print("\nStep 2: Processing financial data...")
        try:
            
            data_processor = DataProcessingAgent(model=None)
            financial_data = data_processor.execute(raw_data=financial_data)
            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
    
    # Create Dashboard
    if not args.no_dashboard:
        print("\nStep 3: Creating financial dashboard...")
        try:
            #  model=None
            visualizer = VisualizationAgent(model=None)
            dashboard = visualizer.execute(data=financial_data)
            logger.info("Dashboard created successfully")
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            dashboard = None
    else:
        dashboard = None
    
    # Profit Forecasting
    if not args.skip_forecasting and not args.dashboard_only:
        print("\nStep 4: Generating profit forecasts...")
        try:
            # Initialize with model=None
            forecaster = ForecastingAgent(model=None)
            forecasts = forecaster.execute(historical_data=financial_data)
            logger.info("Profit forecasting completed successfully")
        except Exception as e:
            logger.error(f"Error during profit forecasting: {str(e)}")
    
    # Generate summary report
    if not args.dashboard_only:
        print("\nGenerating analysis summary report...")
        generate_summary_report(financial_data)
    
    # Launch dashboard if available
    if dashboard and not args.no_dashboard:
        print("\nAnalysis complete! Starting dashboard...")
        print("Dashboard will be available at http://127.0.0.1:8050/")
        print("Press Ctrl+C to exit")
        
        
        dashboard.run(debug=False)
    else:
        print("\nAnalysis complete! Results saved to the 'output' directory.")

def generate_summary_report(data):
    """Generate a summary report of the financial analysis"""
    os.makedirs('output/reports', exist_ok=True)
    
    # Create report content
    report = f"""# CSE Financial Analysis - Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Companies Analyzed
- Dipped Products PLC (DIPD.N0000)
- Richard Pieris Exports PLC (REXP.N0000)

## Data Overview

"""
    
    # Add data overview for each company
    for company in ["DIPD.N0000", "REXP.N0000"]:
        df = data.get(company)
        if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
            # Convert to DataFrame if not already
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                else:
                    logger.warning(f"Could not set index for {company}")
                    continue
            
            report += f"### {company}\n"
            report += f"- Periods analyzed: {len(df)} quarters\n"
            
            # Safely get date range
            try:
                report += f"- Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}\n"
            except Exception as e:
                logger.warning(f"Could not get date range for {company}: {e}")
                report += "- Date range: Unable to determine\n"
            
            # Financial metrics
            report += "\n#### Key Financial Metrics (Average)\n"
            metrics_to_check = ['Revenue', 'Gross Profit', 'Gross Margin', 
                               'Operating Income', 'Net Income', 'Net Margin']
            
            for metric in metrics_to_check:
                if metric in df.columns:
                    try:
                        if 'Margin' in metric:
                            report += f"- {metric}: {df[metric].mean():.2f}%\n"
                        else:
                            report += f"- {metric}: {df[metric].mean():,.2f}\n"
                    except Exception as e:
                        logger.warning(f"Could not calculate {metric} for {company}: {e}")
            
            # Growth metrics
            report += "\n#### Growth Metrics (Average)\n"
            growth_metrics = ['Revenue_QoQ_Growth', 'Net_Income_QoQ_Growth']
            
            for metric in growth_metrics:
                if metric in df.columns:
                    try:
                        report += f"- {metric.replace('_', ' ')}: {df[metric].mean():.2f}%\n"
                    except Exception as e:
                        logger.warning(f"Could not calculate {metric} for {company}: {e}")
            
            report += "\n"
    
    # Add comparative analysis
    report += """## Comparative Analysis
For detailed comparative analysis, please refer to the interactive dashboard.

## Forecast Results
Profit forecasts for future quarters are available in the 'profit_forecast_report.md' file.

## Files Generated
- Processed data: data/processed/
- Visualizations: output/visualizations/
- Forecast charts: output/forecasts/
- Reports: output/reports/

## Dashboard
An interactive dashboard is available for exploring the financial data.
To launch the dashboard, run: python main.py --dashboard-only
"""
    
    # Write report to file
    with open('output/reports/analysis_summary.md', 'w') as f:
        f.write(report)
    
    logger.info("Summary report generated: output/reports/analysis_summary.md")

if __name__ == "__main__":
    main()