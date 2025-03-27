# CSE Financial Analysis

## Project Overview
This tool analyzes quarterly financial data from two companies listed on the Colombo Stock Exchange (CSE): Dipped Products PLC (DIPD.N0000) and Richard Pieris Exports PLC (REXP.N0000). It extracts data from PDF reports, creates structured datasets, visualizes trends through an interactive dashboard, and forecasts future performance.

## Environment Setup

### Prerequisites
- Python 3.8 or higher

### Installation Steps

1. **Clone or download the project**
   ```
   git clone https://github.com/Kusalani/Quarterly-Financial-Analysis.git
   cd cse-financial-analysis
   ```
   
   If you received the project as a ZIP file, extract it and navigate to the directory.

2. **Create a virtual environment (recommended)**
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   
   

## Running the Program

### Full Analysis Pipeline

To run the complete analysis (scraping, processing, visualization, and forecasting):

```
python main.py
```

This will:
1. Download and extract data from quarterly reports
2. Process the data into structured datasets
3. Generate forecasts for future quarters
4. Launch the interactive dashboard

### Command-Line Options

The program supports several command-line options for more control:

```
python main.py [options]
```

Available options:
- `--skip-scraping`: Skip the data scraping step (use previously downloaded data)
- `--skip-processing`: Skip the data processing step (use pre-processed data)
- `--skip-forecasting`: Skip the profit forecasting step
- `--dashboard-only`: Run only the dashboard using existing data
- `--no-dashboard`: Run without launching the dashboard

Examples:
```
# Run only the dashboard with existing data
python main.py --dashboard-only

# Update forecasts with existing data and show dashboard
python main.py --skip-scraping --skip-processing

# Process data and generate reports without showing dashboard
python main.py --skip-scraping --no-dashboard
```

## Accessing the Dashboard

Once the program is running with the dashboard enabled:

1. Open a web browser
2. Go to: http://127.0.0.1:8050/
3. The dashboard will load automatically

### Dashboard Features

The dashboard includes:
- Company selector: Choose between DIPD, REXP, or comparative view
- Date range selector: Filter data by specific time periods
- Metrics selector: Choose which financial metrics to display
- Interactive charts showing:
  - Financial performance over time
  - Margin analysis
  - Growth rates quarter-over-quarter
  - Company comparisons

## Project Structure

```
cse-financial-analysis/
├── main.py                      # Main program
├── dipd_financial_scraper.py    # DIPD data extraction
├── rexp_financial_scraper.py    # REXP data extraction
├── agents/
│   ├── data_processing_agent.py # Data processing
│   ├── visualization_agent.py   # Dashboard creation
│   └── forecasting_agent.py     # Forecasting
├── forecast_profits.py          # Forecasting algorithms
├── data/                        # Data directory
│   ├── extracted/               # Raw data
│   └── processed/               # Processed datasets
├── output/                      # Output files
│   ├── forecasts/               # Forecast results
│   ├── reports/                 # Generated reports
│   └── visualizations/          # Dashboard screenshots
└── requirements.txt             # Python dependencies
```

## Troubleshooting

- **Missing data folders**: The program will create necessary directories automatically
- **PDF download errors**: Check your internet connection; the program will retry failed downloads
- **Dashboard not loading**: Make sure port 8050 is not being used by another application
- **Forecasting errors**: If forecasting fails, try running with `--skip-forecasting` option

If issues persist, check the console output for error messages or contact the developer.

## Notes

- The first run will take longer as it needs to download and process all PDF reports
- Processed data is cached to disk for faster subsequent runs
- Forecast results are saved to `output/reports/profit_forecast_report.md`
- Dashboard screenshots are saved to `output/visualizations/`