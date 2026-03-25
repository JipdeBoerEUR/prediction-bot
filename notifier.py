"""
The "Notifier" - Sends email alerts for manual execution on Trading 212.
Goal: When a stock passes all screens, alert the user via email with exact entry, stop loss, and take profit prices.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Ensure you set these in your environment, e.g., using a Gmail App Password
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "your_email@gmail.com")

def send_trade_alert(ticker: str, title: str, value: str, current_price: float, shares: int, allocated: float, take_profit_pct: float, stop_loss_pct: float, rag_report: str = ""):
    
    # Calculate exact Trading 212 bracket prices
    stop_loss_price = current_price * (1.0 - stop_loss_pct)
    take_profit_price = current_price * (1.0 + take_profit_pct)
    
    subject = f"TRADE ALERT: {ticker} Insider Buy Passed Screens!"
    
    rag_section = f"\n--- AI RISK AUDIT (RAG Library) ---\n{rag_report}\n" if rag_report else ""
    
    body = f"""Prediction Bot has found a stock that passed all Fundamental and Valuation screens!
{rag_section}
--- INSIDER CONVICTION ---
Ticker: {ticker}
Insider: {title}
Value Bought: {value}

--- TRADING 212 EXECUTION PLAN ---
Action: BUY MARKET
Shares Needed: {shares}
Capital Required: ${allocated:,.2f}
Current Price: ${current_price:,.2f}

--- RISK MANAGEMENT BRACKETS ---
Stop Loss (Sell Stop): ${stop_loss_price:,.2f} (-{stop_loss_pct*100}%)
Take Profit (Limit Sell): ${take_profit_price:,.2f} (+{take_profit_pct*100}%)

Execute this trade manually on your Trading 212 app by setting a Market Buy for {shares} shares, and immediately attaching the Stop Loss and Take Profit orders as shown above.
"""

    print(f"\n--- ALERT GENERATED FOR {ticker} ---")
    print(body)

    if SENDER_EMAIL == "your_email@gmail.com":
        print("Notice: Email not actually sent. Please set SENDER_EMAIL, SENDER_PASSWORD, and RECEIVER_EMAIL environment variables to activate SMTP delivery.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to Gmail SMTP (Change 'smtp.gmail.com' if using Yahoo/Outlook)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Success! Trade alert email successfully sent to {RECEIVER_EMAIL}.")
        
    except Exception as e:
        print(f"Failed to send email alert: {e}")


def send_statarb_alert(orders_df):
    """
    Sends an email alert detailing the generated Statistical Arbitrage (pricing mismatch) trades.
    """
    if orders_df is None or orders_df.empty:
        return
        
    subject = "📉📈 STAT-ARB ALERT: Pricing Mismatches Detected!"
    
    body = "Prediction Bot Topological Stat-Arb Engine has found new pricing mismatches and submitted the following trades:\n\n"
    body += "--- SUGGESTED PORTFOLIO REBALANCING ---\n"
    
    for _, row in orders_df.iterrows():
        action = row.get('side', '')
        ticker = row.get('ticker', '')
        qty = row.get('Qty', 0)
        residual = row.get('residual', 0.0)
        
        if action == "BUY":
            body += f"🟢 BUY {qty} shares of {ticker} (Undervalued | Residual: {residual:.3f})\n"
        elif action == "SELL":
            body += f"🔴 SELL {qty} shares of {ticker} (Overvalued/Mean-Reversion | Residual: {residual:.3f})\n"
            
    body += "\nThese trades have been submitted to Alpaca automatically (unless in Dry Run mode).\nCheck your Alpaca dashboard for execution status."
    
    print("\n--- STAT-ARB ALERT GENERATED ---")
    print(body)

    if SENDER_EMAIL == "your_email@gmail.com":
        print("Notice: Stat-Arb email not actually sent. Please set SMTP environment variables.")
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Success! Stat-Arb alert email successfully sent to {RECEIVER_EMAIL}.")
    except Exception as e:
        print(f"Failed to send Stat-Arb email alert: {e}")

if __name__ == "__main__":
    send_trade_alert("AAPL", "CEO", "$150,000", 150.00, 10, 1500.0, 0.20, 0.08, rag_report="Filings appear standard. No material debt issues or active lawsuits found.")
