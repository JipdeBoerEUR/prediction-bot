"""
The "Ledger" - Logs all evaluated trades to a central CSV database.
"""
import csv
import os
from datetime import datetime

LOG_FILE = "trade_history.csv"

def log_evaluation(ticker: str, insider_value: float, signal: str, reasons: str):
    file_exists = os.path.isfile(LOG_FILE)
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write the header row
                writer.writerow(['Date', 'Ticker', 'Insider_Value', 'Signal', 'Reasons'])
                
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker,
                insider_value,
                signal,
                reasons
            ])
    except Exception as e:
        print(f"Warning: Failed to log evaluation for {ticker}: {e}")
