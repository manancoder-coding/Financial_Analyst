import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph
from langchain_core.pydantic_v1 import BaseModel
import groq  # Using Groq API instead of OpenAI

# Set Groq API Key
GROQ_API_KEY = "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"
client = groq.Client(api_key=GROQ_API_KEY)

# Define State Schema for LangGraph
class StockAnalysisState(BaseModel):
    ticker: str
    data: dict = None
    analysis: str = None
    financial_metrics: dict = None

def fetch_stock_data(state: StockAnalysisState):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(state.ticker)
    hist = stock.history(period="6mo")
    state.data = hist.to_dict()
    state.financial_metrics = stock.info
    return state

def compute_technical_indicators(state: StockAnalysisState):
    """Computes RSI, MACD, Moving Averages, and VWAP."""
    data = pd.DataFrame(state.data)
    
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    
    state.data = data.to_dict()
    return state

def ai_analyze_stock(state: StockAnalysisState):
    """Uses Groq API to analyze stock data."""
    data = pd.DataFrame(state.data)
    latest = data.iloc[-1]
    
    financials = state.financial_metrics
    pe_ratio = financials.get('trailingPE', 'N/A')
    debt_to_equity = financials.get('debtToEquity', 'N/A')
    profit_margin = financials.get('profitMargins', 'N/A')
    
    analysis_prompt = f""" 
    You are a financial analyst providing expert stock analysis.
    Analyze the stock {state.ticker} based on the following metrics:

    Technical Analysis:
    - Current Price: {latest['Close']}
    - 50-Day Moving Average: {latest['MA50']}
    - 200-Day Moving Average: {latest['MA200']}
    - RSI: {latest['RSI']} (Overbought >70, Oversold <30)
    - VWAP: {latest['VWAP']} (Compare to Current Price)
    - MACD: {latest['MACD']} (Bullish if >0, Bearish if <0)

    Fundamental Analysis:
    - P/E Ratio: {pe_ratio} (Higher = Expensive, Lower = Cheap)
    - Debt-to-Equity: {debt_to_equity} (Higher = More Debt Risk)
    - Profit Margins: {profit_margin} (Higher = More Efficient)

    Provide insights and classify the stock as a **Buy, Hold, or Sell** recommendation.
    """
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": analysis_prompt},
        ],
    )

    state.analysis = response.choices[0].message.content
    return state

# LangGraph Workflow
graph = StateGraph(state_schema=StockAnalysisState)

graph.add_node("fetch_data", fetch_stock_data)
graph.add_node("compute_indicators", compute_technical_indicators)
graph.add_node("ai_analysis", ai_analyze_stock)

graph.set_entry_point("fetch_data")
graph.add_edge("fetch_data", "compute_indicators")
graph.add_edge("compute_indicators", "ai_analysis")

graph = graph.compile()

# Streamlit UI
def main():
    st.title("📈 Financial Analyst AI")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG)")

    # Initialize session state variables if not present
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "approved" not in st.session_state:
        st.session_state.approved = False

    if st.button("Analyze Stock") and ticker:
        state = StockAnalysisState(ticker=ticker.upper())
        result = graph.invoke(state)

        # Store analysis in session state to persist across reruns
        st.session_state.analysis = result.get("analysis", "No analysis available")
        st.session_state.approved = False  # Reset approval state for new analysis

    # Show analysis only if it has been generated
    if st.session_state.analysis:
        st.subheader("🔍 AI-Generated Financial Analysis")
        st.write(st.session_state.analysis)

        # 📊 Display Stock Chart
        state = StockAnalysisState(ticker=ticker.upper())
        fetch_stock_data(state)
        compute_technical_indicators(state)

        data = pd.DataFrame(state.data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Close'], label="Stock Price", color='blue', linewidth=2)
        ax.plot(data.index, data['MA50'], label="50-Day MA", color='orange', linestyle='dashed')
        ax.plot(data.index, data['MA200'], label="200-Day MA", color='red', linestyle='dashed')

        ax.set_title(f"{state.ticker} Stock Price & Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # 📊 Display MACD Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['MACD'], label="MACD", color='purple')
        ax.axhline(0, color='black', linestyle='dashed')

        ax.set_title(f"{state.ticker} MACD Indicator")
        ax.legend()
        st.pyplot(fig)

        # Approval button
        if st.button("Approve AI Analysis", key="approve_button"):
            st.session_state.approved = True

    # Display final report only if approved
    if st.session_state.approved:
        st.subheader("✅ Final Financial Report")
        st.write(st.session_state.analysis)

if __name__ == "__main__":
    main()
