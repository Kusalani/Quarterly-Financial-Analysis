import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_processing_agent')

class DataProcessingAgent:
    """
    Agent responsible for processing and structuring raw financial data
    """
    def __init__(self, model=None):
        """Initialize the data processing agent"""
        self.model = model  # Can be None, we don't use it for processing
        logger.info("Data Processing Agent initialized")
    
    def execute(self, raw_data=None):
        """
        Process raw financial data into structured format
        
        Args:
            raw_data (dict): Dictionary of DataFrames with company-specific financial data
            
        Returns:
            dict: Processed and structured data
        """
        logger.info("Processing financial data...")
        
        try:
            # Process raw JSON files directly
            processed_data = self._process_raw_json_files()
            
            # If no data was processed from JSON files but raw_data is provided, use it
            if all(df.empty for df in processed_data.values()) and raw_data is not None:
                processed_data = {
                    company: df.copy() for company, df in raw_data.items() if not df.empty
                }
                
                # Process each company's data
                for company, df in processed_data.items():
                    if company != 'combined':
                        logger.info(f"Processing data for {company}")
                        processed_data[company] = self._process_company_data(df, company)
                        logger.info(f"Completed processing for {company}")
            
            # Create combined dataset if needed
            if 'combined' not in processed_data or processed_data['combined'].empty:
                company_dfs = [df for company, df in processed_data.items() 
                              if company != 'combined' and not df.empty]
                if company_dfs:
                    processed_data['combined'] = pd.concat(company_dfs)
                    processed_data['combined'].sort_index(inplace=True)
                    logger.info("Created combined dataset")
            
            # Save processed data
            self._save_processed_data(processed_data)
            
            logger.info("Data processing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _process_raw_json_files(self):
        """
        Process raw JSON files directly
        
        Returns:
            dict: Dictionary of processed DataFrames
        """
        processed_data = {}
        
        # Process company-specific data
        for company in ["DIPD.N0000", "REXP.N0000"]:
            json_path = f'data/extracted/{company}_raw.json'
            
            if os.path.exists(json_path):
                try:
                    logger.info(f"Processing raw JSON for {company}")
                    df = self._convert_json_to_dataframe(json_path, company)
                    processed_data[company] = df
                    logger.info(f"Successfully processed {len(df)} records for {company}")
                except Exception as e:
                    logger.error(f"Error processing JSON for {company}: {str(e)}")
                    processed_data[company] = pd.DataFrame()
            else:
                logger.warning(f"Raw JSON not found for {company}: {json_path}")
                processed_data[company] = pd.DataFrame()
        
        # Create combined dataset
        if all(not df.empty for df in [processed_data.get("DIPD.N0000", pd.DataFrame()), 
                                      processed_data.get("REXP.N0000", pd.DataFrame())]):
            processed_data['combined'] = pd.concat([
                processed_data["DIPD.N0000"], 
                processed_data["REXP.N0000"]
            ])
            processed_data['combined'].sort_index(inplace=True)
            logger.info(f"Created combined dataset with {len(processed_data['combined'])} records")
        else:
            processed_data['combined'] = pd.DataFrame()
        
        return processed_data
    
    def _convert_json_to_dataframe(self, json_path, company_name):
        """
        Convert raw JSON financial data to DataFrame with minimal transformation
        
        Args:
            json_path (str): Path to the JSON file
            company_name (str): Name of the company
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Load JSON data
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to list of records
        records = []
        
        for quarter_key, quarter_data in raw_data.items():
            # Extract year and quarter from the key (e.g., "2022-Q1")
            try:
                year_str, quarter_str = quarter_key.split('-Q')
                year = int(year_str)
                quarter = int(quarter_str)
            except Exception as e:
                logger.warning(f"Skipping invalid quarter key: {quarter_key}")
                continue
            
            # Create proper date (first day of the quarter)
            date = datetime(year=year, month=((quarter-1)*3)+1, day=1)
            
            # Create a record with all available data
            record = {
                'Date': date,
                'Company': company_name,
                'Year': year,
                'Quarter': quarter
            }
            
            # Add all financial metrics
            for key, value in quarter_data.items():
                if key not in ['Company', 'Year', 'Quarter', 'URL']:
                    record[key] = value
            
            records.append(record)
        
        # Create DataFrame
        if not records:
            logger.warning(f"No records found in JSON file for {company_name}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Set Date as index
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        
        # Calculate margin ratios where possible
        self._calculate_simple_margins(df)
        
        return df
    
    def _calculate_simple_margins(self, df):
        """
        Calculate simple margin ratios without complex transformations
        
        Args:
            df (pd.DataFrame): DataFrame to process
        """
        # Gross Margin
        if 'Gross Profit' in df.columns and 'Revenue' in df.columns:
            # Only calculate where both values are present and Revenue is positive
            valid_rows = (df['Gross Profit'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0))
            if valid_rows.any():
                df.loc[valid_rows, 'Gross Margin'] = (df.loc[valid_rows, 'Gross Profit'] / df.loc[valid_rows, 'Revenue'] * 100)
        
        # Operating Margin
        if 'Operating Income' in df.columns and 'Revenue' in df.columns:
            # Only calculate where both values are present and Revenue is positive
            valid_rows = (df['Operating Income'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0))
            if valid_rows.any():
                df.loc[valid_rows, 'Operating Margin'] = (df.loc[valid_rows, 'Operating Income'] / df.loc[valid_rows, 'Revenue'] * 100)
        
        # Net Margin
        if 'Net Income' in df.columns and 'Revenue' in df.columns:
            # Only calculate where both values are present and Revenue is positive
            valid_rows = (df['Net Income'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0))
            if valid_rows.any():
                df.loc[valid_rows, 'Net Margin'] = (df.loc[valid_rows, 'Net Income'] / df.loc[valid_rows, 'Revenue'] * 100)
    
    def _process_company_data(self, df, company_name):
        """
        Process a company's financial data with minimal transformation
        
        Args:
            df (pd.DataFrame): Raw financial data
            company_name (str): Name of the company for logging
            
        Returns:
            pd.DataFrame: Processed financial data
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle Date index conversion if needed
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            if 'Date' in processed_df.columns:
                processed_df.set_index('Date', inplace=True)
            elif 'Year' in processed_df.columns and 'Quarter' in processed_df.columns:
                # Convert Year and Quarter to proper dates
                processed_df['Date'] = pd.to_datetime([
                    f"{year}-{(quarter-1)*3+1:02d}-01"  # Q1 -> month 1, Q2 -> month 4, etc.
                    for year, quarter in zip(processed_df['Year'], processed_df['Quarter'])
                ])
                processed_df.set_index('Date', inplace=True)
            else:
                logger.warning(f"{company_name}: No valid date information found")
        
        # Ensure numeric columns are the right type
        for col in processed_df.columns:
            if col not in ['Company', 'Year', 'Quarter'] and processed_df[col].dtype == object:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Calculate simple margins
        self._calculate_simple_margins(processed_df)
        
        # Ensure the index is sorted
        processed_df = processed_df.sort_index()
        
        return processed_df
    
    # This function should replace the existing _save_processed_data method in DataProcessingAgent class

    def _save_processed_data(self, processed_data):
        """
        Save processed data to files with improved handling of combined dataset
        
        Args:
            processed_data (dict): Dictionary of processed DataFrames
        """
        # Create the directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save individual company data
        for company in ["DIPD.N0000", "REXP.N0000"]:
            if company in processed_data and not processed_data[company].empty:
                # Make sure Company column exists
                df = processed_data[company].copy()
                if "Company" not in df.columns:
                    df["Company"] = company
                
                # Save as CSV
                df.to_csv(f'data/processed/{company}_processed.csv')
                
                # Save as pickle
                df.to_pickle(f'data/processed/{company}_processed.pkl')
                
                logger.info(f"Saved processed data for {company}: {len(df)} rows")
        
        # Create and save combined dataset if both companies have data
        if ("DIPD.N0000" in processed_data and not processed_data["DIPD.N0000"].empty and
            "REXP.N0000" in processed_data and not processed_data["REXP.N0000"].empty):
            
            # Create copies to avoid modifying originals
            dipd_df = processed_data["DIPD.N0000"].copy()
            rexp_df = processed_data["REXP.N0000"].copy()
            
            # Ensure both have Company column
            if "Company" not in dipd_df.columns:
                dipd_df["Company"] = "DIPD.N0000"
            if "Company" not in rexp_df.columns:
                rexp_df["Company"] = "REXP.N0000"
            
            # Combine the data
            combined_df = pd.concat([dipd_df, rexp_df])
            
            # Sort by date
            if isinstance(combined_df.index, pd.DatetimeIndex):
                combined_df = combined_df.sort_index()
            
            # Save combined dataset
            combined_df.to_csv('data/processed/combined_data.csv')
            combined_df.to_pickle('data/processed/combined_data.pkl')
            
            # Add to processed_data dictionary
            processed_data['combined'] = combined_df
            
            # Log company counts for verification
            dipd_count = (combined_df["Company"] == "DIPD.N0000").sum()
            rexp_count = (combined_df["Company"] == "REXP.N0000").sum()
            logger.info(f"Saved combined data with {len(combined_df)} rows: DIPD={dipd_count}, REXP={rexp_count}")