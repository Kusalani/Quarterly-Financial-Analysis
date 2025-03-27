# forecast_profits.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

def forecast_financial_data(historical_data, periods=4):
    """
    Generate forecasts for the financial data.
    
    Args:
        historical_data (dict): Dictionary of DataFrames with historical financial data
        periods (int): Number of future periods to forecast
        
    Returns:
        dict: Dictionary containing forecast results for each company
    """
    print("Starting profit forecasting...")
    
    # Create directories for output
    os.makedirs('output/forecasts', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    
    warnings.filterwarnings('ignore')  # Suppress statsmodels warnings
    
    forecast_results = {}
    
    # Get company-specific data
    companies = [key for key in historical_data.keys() if key != 'combined']
    
    for company in companies:
        print(f"Generating forecasts for {company}")
        
        # Get company data
        company_data = historical_data[company]
        
        # Only forecast if we have enough data
        if len(company_data) < 8:
            print(f"Not enough data for {company} to generate reliable forecasts. Need at least 8 quarters.")
            continue
        
        # Generate forecasts for different metrics
        metrics = ['Revenue', 'Net Income', 'Gross Profit', 'Operating Income']
        company_forecasts = {}
        
        for metric in metrics:
            if metric not in company_data.columns:
                print(f"Metric {metric} not found in data for {company}")
                continue
                
            print(f"Forecasting {metric} for {company}")
            
            # Get historical values for the metric
            historical_values = company_data[metric]
            
            # Prepare data for forecasting
            data = pd.Series(historical_values.values, index=company_data.index)
            
            # Fill any missing values
            data = data.interpolate()
            
            # Apply forecasting models
            forecast_result = apply_best_forecasting_model(data, periods=periods)
            
            # Add forecast to results
            company_forecasts[metric] = forecast_result
            
            # Create visualization
            create_forecast_visualization(company, metric, data, forecast_result)
        
        forecast_results[company] = company_forecasts
    
    # Generate report
    generate_forecast_report(companies, historical_data, forecast_results)
    
    print("Forecasting completed successfully!")
    
    return forecast_results

def apply_best_forecasting_model(data, periods=4):
    """
    Apply multiple forecasting models and select the best one.
    
    Args:
        data (pd.Series): Time series data to forecast
        periods (int): Number of periods to forecast
        
    Returns:
        dict: Dictionary with forecast results
    """
    models = []
    
    # Try exponential smoothing
    try:
        # Configure the model (with seasonal period of 4 for quarterly data)
        hw_model = ExponentialSmoothing(
            data, 
            seasonal_periods=4,
            trend='add',
            seasonal='add',
            use_boxcox=False,
            initialization_method='estimated'
        )
        
        # Fit the model
        hw_fit = hw_model.fit(optimized=True, remove_bias=True)
        
        # Generate forecast
        hw_forecast = hw_fit.forecast(steps=periods)
        
        # Calculate prediction intervals
        residuals_std = hw_fit.resid.std()
        hw_lower = hw_forecast - 1.96 * residuals_std
        hw_upper = hw_forecast + 1.96 * residuals_std
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((data - hw_fit.fittedvalues) / data)) * 100
        
        models.append({
            'name': 'Holt-Winters',
            'mape': mape,
            'forecast': hw_forecast,
            'lower': hw_lower,
            'upper': hw_upper,
            'model': hw_fit
        })
    except Exception as e:
        print(f"Error with Holt-Winters model: {str(e)}")
    
    # Try ARIMA model
    try:
        arima_model = ARIMA(data, order=(2,1,2))
        arima_fit = arima_model.fit()
        
        # Generate forecast
        arima_forecast = arima_fit.forecast(steps=periods)
        
        # Get prediction intervals
        pred_int = arima_fit.get_forecast(steps=periods).conf_int()
        arima_lower = pred_int.iloc[:, 0]
        arima_upper = pred_int.iloc[:, 1]
        
        # Calculate MAPE
        mape = np.mean(np.abs((data - arima_fit.fittedvalues) / data)) * 100
        
        models.append({
            'name': 'ARIMA',
            'mape': mape,
            'forecast': arima_forecast,
            'lower': arima_lower,
            'upper': arima_upper,
            'model': arima_fit
        })
    except Exception as e:
        print(f"Error with ARIMA model: {str(e)}")
    
    # If all models failed, use simple average growth method
    if not models:
        print("All sophisticated models failed. Using simple growth method.")
        
        # Calculate average growth rate
        pct_changes = data.pct_change().dropna()
        avg_growth_rate = pct_changes.mean()
        
        # Generate forecast
        forecast_values = []
        last_value = data.iloc[-1]
        
        for i in range(periods):
            next_value = last_value * (1 + avg_growth_rate)
            forecast_values.append(next_value)
            last_value = next_value
        
        # Create forecast series
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            freq='QS',
            periods=periods
        )
        forecast = pd.Series(forecast_values, index=future_dates)
        
        # Simple confidence intervals (±15%)
        lower = forecast * 0.85
        upper = forecast * 1.15
        
        return {
            'model_name': 'Simple Growth',
            'forecast': forecast,
            'lower_bound': lower,
            'upper_bound': upper,
            'mape': np.nan
        }
    
    # Select best model based on MAPE
    best_model = min(models, key=lambda x: x['mape'])
    print(f"Selected {best_model['name']} model with MAPE of {best_model['mape']:.2f}%")
    
    # Create forecast dates
    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        freq='QS',
        periods=periods
    )
    
    # Return forecast result
    return {
        'model_name': best_model['name'],
        'forecast': best_model['forecast'],
        'lower_bound': best_model['lower'],
        'upper_bound': best_model['upper'],
        'mape': best_model['mape']
    }

def create_forecast_visualization(company, metric, historical, forecast_result):
    """
    Create visualization of historical data and forecast.
    
    Args:
        company (str): Company name
        metric (str): Metric name
        historical (pd.Series): Historical values
        forecast_result (dict): Dictionary with forecast results
    """
    plt.figure(figsize=(10, 6))
    
    # Plot historical data
    plt.plot(historical.index, historical.values, 'b-', label='Historical')
    
    # Plot forecast
    forecast = forecast_result['forecast']
    lower = forecast_result['lower_bound']
    upper = forecast_result['upper_bound']
    
    plt.plot(forecast.index, forecast.values, 'r--', label='Forecast')
    
    # Plot confidence interval
    plt.fill_between(
        forecast.index,
        lower.values,
        upper.values,
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.title(f'{company} - {metric} Forecast ({forecast_result["model_name"]} Model)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save figure
    os.makedirs('output/forecasts/plots', exist_ok=True)
    filename = f"{company.replace('.', '_')}_{metric.replace(' ', '_')}_forecast.png"
    plt.savefig(f'output/forecasts/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def generate_forecast_report(companies, historical_data, forecast_results):
    """
    Generate a comprehensive forecast report.
    
    Args:
        companies (list): List of company names
        historical_data (dict): Dictionary with historical data for each company
        forecast_results (dict): Dictionary with forecast results for each company
        
    Returns:
        str: Path to the generated report
    """
    print("Generating forecast report...")
    
    # Create report header
    report = "# Profit Forecasting: Methodology and Results\n\n"
    report += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add methodology section
    report += "## Methodology\n\n"
    report += "This profit forecasting analysis employs multiple time series forecasting models to predict future financial performance. "
    report += "The forecasting process involves the following steps:\n\n"
    report += "1. **Data Preparation**: Historical quarterly financial data was prepared and cleaned.\n"
    report += "2. **Model Selection**: Multiple forecasting models were applied to each metric:\n"
    report += "   - Holt-Winters Exponential Smoothing (with additive seasonality)\n"
    report += "   - ARIMA (AutoRegressive Integrated Moving Average)\n"
    report += "3. **Model Evaluation**: Models were evaluated using Mean Absolute Percentage Error (MAPE) on historical data.\n"
    report += "4. **Best Model Selection**: The model with the lowest MAPE was selected for forecasting.\n"
    report += "5. **Forecast Generation**: The selected model was used to generate forecasts for the next 4 quarters.\n"
    report += "6. **Confidence Intervals**: 95% confidence intervals were calculated to indicate forecast uncertainty.\n\n"
    
    # Add key assumptions
    report += "## Key Assumptions\n\n"
    report += "The forecasting models make several key assumptions:\n\n"
    report += "1. **Pattern Continuity**: Historical patterns in the data will continue into the future.\n"
    report += "2. **Seasonality**: Quarterly seasonality patterns will remain consistent.\n"
    report += "3. **No Structural Changes**: No major changes in company structure, operations, or market conditions.\n"
    report += "4. **Economic Stability**: Overall economic conditions will remain relatively stable.\n\n"
    
    # Add company forecast results
    report += "## Forecast Results\n\n"
    
    for company in companies:
        if company not in forecast_results:
            continue
            
        report += f"### {company}\n\n"
        
        # Add forecast results table
        report += "#### Financial Metric Forecasts (Next 4 Quarters)\n\n"
        report += "| Metric | Model | Next Quarter | Q+2 | Q+3 | Q+4 | Avg. Growth |\n"
        report += "|--------|-------|-------------|-----|-----|-----|-------------|\n"
        
        # For each metric
        for metric, forecast_result in forecast_results[company].items():
            forecast = forecast_result['forecast']
            last_historical = historical_data[company][metric].iloc[-1]
            
            # Format forecast values
            forecast_values = []
            growth_rates = []
            
            for i, value in enumerate(forecast):
                forecast_values.append(f"{value:,.2f}")
                
                # Calculate growth rate
                if i == 0:
                    growth = ((value / last_historical) - 1) * 100
                else:
                    growth = ((value / forecast.iloc[i-1]) - 1) * 100
                
                growth_rates.append(growth)
            
            # Calculate average growth
            avg_growth = np.mean(growth_rates)
            
            # Add to report
            report += f"| {metric} | {forecast_result['model_name']} | {forecast_values[0]} | {forecast_values[1]} | {forecast_values[2]} | {forecast_values[3]} | {avg_growth:.2f}% |\n"
        
        # Add model performance
        report += "\n#### Model Performance\n\n"
        report += "| Metric | Selected Model | MAPE |\n"
        report += "|--------|---------------|------|\n"
        
        for metric, forecast_result in forecast_results[company].items():
            mape = forecast_result['mape']
            mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
            
            report += f"| {metric} | {forecast_result['model_name']} | {mape_str} |\n"
        
        # Add visualization references
        report += "\n#### Forecast Visualizations\n\n"
        
        for metric in forecast_results[company].keys():
            filename = f"{company.replace('.', '_')}_{metric.replace(' ', '_')}_forecast.png"
            report += f"![{company} {metric} Forecast](plots/{filename})\n\n"
        
        report += "\n"
    
    # Add limitations
    report += "## Limitations and Caveats\n\n"
    report += "While the forecasting models provide valuable insights, some limitations should be considered:\n\n"
    report += "1. **Limited Historical Data**: The models are based on a limited historical dataset.\n"
    report += "2. **Uncertainty**: All forecasts involve inherent uncertainty, as indicated by the confidence intervals.\n"
    report += "3. **External Factors**: The models cannot account for unpredictable external events.\n"
    report += "4. **Market Changes**: Significant changes in market conditions may invalidate the forecasts.\n\n"
    
    # Write report to file
    report_path = 'output/reports/profit_forecast_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Forecast report saved to {report_path}")
    return report_path