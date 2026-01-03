# Custom Charts - Streamlit Deployment

This is a standalone version of the Custom Charts feature for Streamlit Cloud hosting.

## Requirements

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- CBBD API library

## Setup for Streamlit Cloud

1. **Push to GitHub**: 
   - Create a new repository or use existing one
   - Add these files:
     - `custom_charts_app.py` (the main app)
     - `requirements.txt`
     - `README.md` (this file)

2. **Streamlit Cloud Secrets**:
   - Go to your app settings on Streamlit Cloud
   - Add your CBBD API key to secrets:
     ```toml
     CBBD_API_KEY = "your-api-key-here"
     ```

3. **Deploy**:
   - Point Streamlit Cloud to your repository
   - Set main file path to `custom_charts_app.py`
   - Deploy!

## Local Development

For local development, you can either:
- Use Streamlit secrets (`.streamlit/secrets.toml`)
- Or place your API key in a file named `api key` in the project root

Run locally:
```bash
streamlit run custom_charts_app.py
```

## Features

- **Single or Dual Team Comparison**: Compare stats between two teams or analyze one team
- **Multiple Metrics**: Points, FG%, rebounds, assists, and more
- **Trend Analysis**: See if teams are improving or declining in recent games
- **Situation Filters**: Home/away, conference/non-conference
- **Interactive Charts**: Hover for details, zoom, pan

## Data Source

Data provided by the [College Basketball Data API](https://api.collegebasketballdata.com)
