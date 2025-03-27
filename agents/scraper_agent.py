from agno.agent import Agent
from agno.models.google import Gemini  # Using Gemini instead of Groq
from agno.embedder.google import GeminiEmbedder  # Using Gemini for embeddings
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json
import re

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # For Gemini model and embeddings

# Define the PDF URLs for quarterly reports
dipd_urls = [
                "https://cdn.cse.lk/cmt/upload_report_file/670_1652953354945.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1660039239762.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1668050502417.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1676281778649.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1684465615363.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1691572631009.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1699524291151.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1707731795368.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1715853221401.03.2024.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1723086100776.06.2024.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1731321532619.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/670_1739179170721.12.2024.pdf"
    # Add more PDF URLs for DIPD.N0000 here
]

rexp_urls = [
                "https://cdn.cse.lk/cmt/upload_report_file/771_1653995188360.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1660564139519.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1668418805462.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1676454189961.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1685527094110.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1692091050740.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1700049312897.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1707996295121.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1717164700029.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1722423990922.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1730893408597.pdf",
                "https://cdn.cse.lk/cmt/upload_report_file/771_1739185871059.pdf"
]

# Function to extract financial data using Gemini
def extract_financial_data(company, urls):
    """
    Extract financial data from PDF reports using Gemini
    """
    print(f"Extracting financial data for {company}...")
    
    # Create the knowledge base from PDF URLs
    knowledge = PDFUrlKnowledgeBase(
        urls=urls,
        vector_db=LanceDb(
            uri=f"tmp/lancedb_{company}",
            table_name=f"{company}_reports",
            search_type=SearchType.hybrid,
            embedder=GeminiEmbedder(id="models/embedding-001"),  # Using Gemini for embeddings
        ),
    )
    
    # Initialize the agent with the knowledge base and Gemini model
    agent = Agent(
        model=Gemini(id="gemini-1.5-pro"),  # Using Google's Gemini Pro model
        description=f"You are a financial data extraction expert for {company}!",
        instructions=[
            "Extract key financial metrics from quarterly reports.",
            "Focus on Revenue, Cost of Goods Sold, Gross Profit, Operating Expenses, Operating Income, and Net Income.",
            "Extract the quarter and year for each report.",
            "Format all extracted data in a structured JSON format."
        ],
        knowledge=knowledge,
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )
    
    # Load the knowledge base
    if agent.knowledge is not None:
        agent.knowledge.load()
    
    # Extract financial data for each quarter
    quarterly_data = {}
    
    # Get consolidated response
    combined_query = """
    Please extract the following financial metrics from the quarterly report:
    1. Quarter and Year of the report
    2. Revenue/Turnover
    3. Cost of Goods Sold/Cost of Sales
    4. Gross Profit
    5. Operating Expenses
    6. Operating Income/Operating Profit
    7. Net Income/Profit for the period
    
    Format your response as a JSON object with these fields plus the report date. 
    If any value is not available, set it to null.
    Use the Group or Consolidated statement when present instead of Company statement.
    """
    
    response = agent.run(combined_query)
    
    # Parse the response to extract JSON data
    try:
        # Look for JSON block in the response
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find a JSON-like structure in the response
            json_text = re.search(r'({[\s\S]*})', response).group(1)
        
        # Parse the JSON data
        data = json.loads(json_text)
        
        # Extract quarter and year
        quarter = data.get('Quarter')
        year = data.get('Year')
        
        if quarter and year:
            quarter_key = f"{year}-Q{quarter}"
            quarterly_data[quarter_key] = {
                'Year': year,
                'Quarter': quarter,
                'Revenue': data.get('Revenue'),
                'Cost of Goods Sold': data.get('Cost of Goods Sold'),
                'Gross Profit': data.get('Gross Profit'),
                'Operating Expenses': data.get('Operating Expenses'),
                'Operating Income': data.get('Operating Income'),
                'Net Income': data.get('Net Income')
            }
            print(f"Extracted data for {quarter_key}")
        else:
            print("Could not extract quarter and year from the response")
    
    except Exception as e:
        print(f"Error parsing agent response: {str(e)}")
        print(f"Raw response: {response}")
    
    return quarterly_data

# Main function to extract data for both companies
def extract_all_data():
    all_data = {}
    
    # Extract data for DIPD.N0000
    dipd_data = extract_financial_data("DIPD.N0000", dipd_urls)
    all_data["DIPD.N0000"] = dipd_data
    
    # Extract data for REXP.N0000
    rexp_data = extract_financial_data("REXP.N0000", rexp_urls)
    all_data["REXP.N0000"] = rexp_data
    
    # Create directories for saving data
    os.makedirs('data/extracted', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save raw data and convert to DataFrames
    for company, quarterly_data in all_data.items():
        with open(f'data/extracted/{company}_raw.json', 'w') as f:
            json.dump(quarterly_data, f, indent=2)
        
        # Convert to DataFrame
        df = convert_to_dataframe(company, quarterly_data)
        df.to_csv(f'data/processed/{company}_processed.csv', index=True)
        df.to_pickle(f'data/processed/{company}_processed.pkl')
        print(f"Saved processed data for {company}")
    
    # Create combined dataset
    all_dfs = []
    for company, quarterly_data in all_data.items():
        df = convert_to_dataframe(company, quarterly_data)
        all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs)
        combined_df.to_csv('data/processed/combined_data.csv', index=True)
        combined_df.to_pickle('data/processed/combined_data.pkl')
        print("Created combined dataset")
    
    return all_data

def convert_to_dataframe(company, quarterly_data):
    """
    Convert quarterly data to DataFrame.
    
    Args:
        company (str): Company symbol
        quarterly_data (dict): Dictionary of quarterly data
        
    Returns:
        pd.DataFrame: DataFrame of financial data
    """
    # Initialize lists to store data
    data_rows = []
    
    # Extract data for each quarter
    for quarter_key, pl_data in quarterly_data.items():
        # Get year and quarter
        year = pl_data.get('Year')
        quarter = pl_data.get('Quarter')
        
        # Skip if missing basic data
        if not year or not quarter:
            continue
            
        # Create date for this quarter (first day of the quarter)
        date = datetime(year, ((quarter - 1) * 3) + 1, 1)
        
        # Create a row dictionary with all data
        row = {'Company': company, 'Date': date, 'Year': year, 'Quarter': quarter}
        
        # Add all metrics from the quarter data
        for metric, value in pl_data.items():
            if metric not in ['Year', 'Quarter']:
                row[metric] = value
        
        # Append to data rows
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # If empty, return empty DataFrame
    if df.empty:
        return df
        
    # Set Date as index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate additional metrics
    if 'Gross Profit' in df.columns and 'Revenue' in df.columns:
        df['Gross Margin'] = (df['Gross Profit'] / df['Revenue']) * 100
    
    if 'Operating Income' in df.columns and 'Revenue' in df.columns:
        df['Operating Margin'] = (df['Operating Income'] / df['Revenue']) * 100
    
    if 'Net Income' in df.columns and 'Revenue' in df.columns:
        df['Net Margin'] = (df['Net Income'] / df['Revenue']) * 100
    
    # Growth rates
    if 'Revenue' in df.columns:
        df['Revenue_QoQ_Growth'] = df['Revenue'].pct_change() * 100
    
    if 'Net Income' in df.columns:
        df['Net_Income_QoQ_Growth'] = df['Net Income'].pct_change() * 100
    
    # Year-over-Year growth (if we have at least 4 quarters)
    if 'Revenue' in df.columns and len(df) >= 4:
        df['Revenue_YoY_Growth'] = (df['Revenue'] / df['Revenue'].shift(4) - 1) * 100
    
    if 'Net Income' in df.columns and len(df) >= 4:
        df['Net_Income_YoY_Growth'] = (df['Net Income'] / df['Net Income'].shift(4) - 1) * 100
    
    # Return the DataFrame
    return df

if __name__ == "__main__":
    # Extract data from PDF reports
    data = extract_all_data()
    print("Extraction complete!")