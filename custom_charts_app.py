"""
Custom Charts - College Basketball Data Visualization
Standalone app for Streamlit hosting
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="CBB Custom Charts",
    page_icon="📊",
    layout="wide"
)

# Import CBBD API
try:
    import cbbd
    from cbbd.api import games_api, stats_api
    from cbbd.configuration import Configuration
    from cbbd.api_client import ApiClient
except ImportError:
    st.error("⚠️ CBBD library not installed. Run: pip install -r requirements.txt")
    st.stop()


@st.cache_resource
def get_api_client():
    """Initialize and cache the CBBD API client"""
    try:
        # Try to load from secrets first (Streamlit Cloud)
        if hasattr(st, 'secrets') and 'CBBD_API_KEY' in st.secrets:
            api_key = st.secrets['CBBD_API_KEY']
        # Fall back to file for local development
        elif os.path.exists('api key'):
            with open('api key', 'r') as f:
                api_key = f.read().strip()
        else:
            st.error("❌ API key not found. Please add CBBD_API_KEY to Streamlit secrets.")
            st.stop()
        
        config = Configuration()
        config.access_token = api_key
        api_client = ApiClient(config)
        return api_client
    except Exception as e:
        st.error(f"Failed to setup API client: {e}")
        st.stop()


@st.cache_data(ttl=3600)
def fetch_team_games(_api_client, team_name, season=2026):
    """Fetch game data for a team"""
    try:
        api_instance = games_api.GamesApi(_api_client)
        games = api_instance.get_games(team=team_name, season=season)
        
        if not games:
            return pd.DataFrame()
        
        games_data = []
        for game in games:
            game_dict = game.to_dict()
            start_date = game_dict.get('start_date')
            game_dict['game_date'] = pd.to_datetime(start_date, errors='coerce') if start_date else pd.NaT
            games_data.append(game_dict)
        
        df = pd.DataFrame(games_data)
        if len(df) > 0 and 'game_date' in df.columns:
            df = df.sort_values('game_date')
        return df
    except Exception as e:
        st.error(f"Error fetching games for {team_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_team_stats(_api_client, team_name, season=2026):
    """Fetch stats data for a team using get_game_teams endpoint"""
    try:
        api_instance = games_api.GamesApi(_api_client)
        stats = api_instance.get_game_teams(team=team_name, season=season)
        
        if not stats:
            return pd.DataFrame()
        
        stats_data = []
        for stat in stats:
            stat_dict = stat.to_dict()
            
            # Extract key fields
            start_date = stat_dict.get('startDate')
            mapped = {
                'game_date': pd.to_datetime(start_date, errors='coerce') if start_date else pd.NaT,
                'team': stat_dict.get('team'),
                'opponent': stat_dict.get('opponent'),
                'is_home': stat_dict.get('isHome'),
                'pace': stat_dict.get('pace'),
            }
            
            # Extract team stats
            if 'teamStats' in stat_dict and stat_dict['teamStats']:
                team_stats = stat_dict['teamStats']
                mapped['points'] = team_stats.get('points', {}).get('total', 0)
                mapped['field_goals_made'] = team_stats.get('fieldGoals', {}).get('made', 0)
                mapped['field_goals_attempted'] = team_stats.get('fieldGoals', {}).get('attempted', 0)
                mapped['field_goal_pct'] = team_stats.get('fieldGoals', {}).get('pct', 0)
                mapped['three_pointers_made'] = team_stats.get('threePointers', {}).get('made', 0)
                mapped['three_pointers_attempted'] = team_stats.get('threePointers', {}).get('attempted', 0)
                mapped['three_point_pct'] = team_stats.get('threePointers', {}).get('pct', 0)
                mapped['free_throws_made'] = team_stats.get('freeThrows', {}).get('made', 0)
                mapped['free_throws_attempted'] = team_stats.get('freeThrows', {}).get('attempted', 0)
                mapped['free_throw_pct'] = team_stats.get('freeThrows', {}).get('pct', 0)
                mapped['rebounds'] = team_stats.get('rebounds', {}).get('total', 0)
                mapped['offensive_rebounds'] = team_stats.get('rebounds', {}).get('offensive', 0)
                mapped['defensive_rebounds'] = team_stats.get('rebounds', {}).get('defensive', 0)
                mapped['assists'] = team_stats.get('assists', 0)
                mapped['steals'] = team_stats.get('steals', 0)
                mapped['blocks'] = team_stats.get('blocks', 0)
                mapped['turnovers'] = team_stats.get('turnovers', 0)
                mapped['fouls'] = team_stats.get('fouls', 0)
            
            # Extract opponent stats
            if 'opponentStats' in stat_dict and stat_dict['opponentStats']:
                opp_stats = stat_dict['opponentStats']
                mapped['opponent_points'] = opp_stats.get('points', {}).get('total', 0)
                mapped['opponent_field_goal_pct'] = opp_stats.get('fieldGoals', {}).get('pct', 0)
                mapped['opponent_three_point_pct'] = opp_stats.get('threePointers', {}).get('pct', 0)
                mapped['opponent_rebounds'] = opp_stats.get('rebounds', {}).get('total', 0)
                mapped['opponent_turnovers'] = opp_stats.get('turnovers', 0)
            
            stats_data.append(mapped)
        
        df = pd.DataFrame(stats_data)
        if len(df) > 0 and 'game_date' in df.columns:
            df = df.sort_values('game_date')
        return df
    except Exception as e:
        st.error(f"Error fetching stats for {team_name}: {e}")
        return pd.DataFrame()


def main():
    """Main application"""
    
    st.title("📊 College Basketball Custom Charts")
    st.markdown("### Visualize team performance trends and compare statistics")
    
    # Initialize API client
    api_client = get_api_client()
    
    st.divider()
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏀 Team 1")
        chart_team = st.text_input("Enter team name:", key="chart_team1", placeholder="e.g., Duke")
        season1 = st.number_input("Season:", min_value=2020, max_value=2026, value=2026, key="season1")
    
    with col2:
        st.subheader("🏀 Team 2 (Optional)")
        chart_team2 = st.text_input("Enter second team to compare:", key="chart_team2", placeholder="Leave blank for single team")
        if chart_team2:
            season2 = st.number_input("Season:", min_value=2020, max_value=2026, value=2026, key="season2")
        else:
            season2 = season1
    
    if not chart_team:
        st.info("👆 Enter a team name to get started")
        st.stop()
    
    # Fetch data
    with st.spinner(f"Fetching data for {chart_team}..."):
        df_games = fetch_team_games(api_client, chart_team, season1)
        df_stats = fetch_team_stats(api_client, chart_team, season1)
    
    if len(df_games) == 0 or len(df_stats) == 0:
        st.error(f"❌ No data found for {chart_team}")
        st.stop()
    
    # Fetch team 2 data if provided
    df_games2 = None
    df_stats2 = None
    if chart_team2:
        with st.spinner(f"Fetching data for {chart_team2}..."):
            df_games2 = fetch_team_games(api_client, chart_team2, season2)
            df_stats2 = fetch_team_stats(api_client, chart_team2, season2)
        
        if len(df_games2) == 0 or len(df_stats2) == 0:
            st.warning(f"⚠️ No data found for {chart_team2}")
            df_games2 = None
            df_stats2 = None
    
    st.success(f"✅ Loaded {len(df_stats)} games for {chart_team}" + 
               (f" and {len(df_stats2)} games for {chart_team2}" if df_stats2 is not None else ""))
    
    st.divider()
    
    # Metric selection
    st.subheader("📈 Select Metric to Chart")
    
    metric_options = {
        "Points Scored": "points",
        "Points Allowed": "opponent_points",
        "Point Differential": "point_diff",
        "Field Goal %": "field_goal_pct",
        "Opponent FG %": "opponent_field_goal_pct",
        "3-Point %": "three_point_pct",
        "Free Throw %": "free_throw_pct",
        "Rebounds": "rebounds",
        "Assists": "assists",
        "Turnovers": "turnovers",
        "Steals": "steals",
        "Blocks": "blocks",
        "Pace": "pace"
    }
    
    selected_metric_label = st.selectbox(
        "Choose metric:",
        options=list(metric_options.keys()),
        index=0
    )
    selected_metric = metric_options[selected_metric_label]
    
    # Situation filter
    situation_filter = st.selectbox(
        "Filter by situation:",
        options=["All Games", "Home Games", "Away Games", "Conference Games", "Non-Conference Games"],
        index=0
    )
    
    # Apply filters
    df_games_filtered = df_stats.copy()
    
    if situation_filter == "Home Games":
        df_games_filtered = df_games_filtered[df_games_filtered['is_home'] == True]
    elif situation_filter == "Away Games":
        df_games_filtered = df_games_filtered[df_games_filtered['is_home'] == False]
    
    if chart_team2 and df_stats2 is not None:
        df_games2_filtered = df_stats2.copy()
        if situation_filter == "Home Games":
            df_games2_filtered = df_games2_filtered[df_games2_filtered['is_home'] == True]
        elif situation_filter == "Away Games":
            df_games2_filtered = df_games2_filtered[df_games2_filtered['is_home'] == False]
    else:
        df_games2_filtered = None
    
    # Prepare data
    df_games_analysis = df_games_filtered.copy()
    df_games2_analysis = df_games2_filtered.copy() if df_games2_filtered is not None else None
    
    if len(df_games_analysis) == 0:
        st.error(f"❌ No games found matching filter: {situation_filter}")
        st.stop()
    
    df_games_analysis = df_games_analysis.sort_values('game_date')
    
    # Calculate point differential if needed
    if selected_metric == "point_diff" and 'point_diff' not in df_games_analysis.columns:
        df_games_analysis['point_diff'] = df_games_analysis['points'] - df_games_analysis['opponent_points']
    
    if df_games2_analysis is not None and len(df_games2_analysis) > 0:
        df_games2_analysis = df_games2_analysis.sort_values('game_date')
        if selected_metric == "point_diff" and 'point_diff' not in df_games2_analysis.columns:
            df_games2_analysis['point_diff'] = df_games2_analysis['points'] - df_games2_analysis['opponent_points']
    
    # Trend Analysis
    if selected_metric in df_games_analysis.columns:
        st.write("**📈 Trend Analysis**")
        trend_cols = st.columns(2 if df_games2_analysis is None else 2)
        
        with trend_cols[0]:
            last_5_avg = df_games_analysis[selected_metric].tail(5).mean()
            season_avg = df_games_analysis[selected_metric].mean()
            trend_diff = last_5_avg - season_avg
            trend_pct = (trend_diff / season_avg * 100) if season_avg != 0 else 0
            
            # Determine if higher is better (offensive) or lower is better (defensive)
            lower_is_better = 'opponent' in selected_metric.lower() or 'allowed' in selected_metric.lower()
            
            if abs(trend_pct) > 5:  # Significant trend
                if lower_is_better:
                    # For defensive stats: lower = improving
                    trend_emoji = "📉" if trend_diff > 0 else "📈"
                    trend_direction = "declining" if trend_diff > 0 else "improving"
                else:
                    # For offensive stats: higher = improving
                    trend_emoji = "📈" if trend_diff > 0 else "📉"
                    trend_direction = "improving" if trend_diff > 0 else "declining"
                st.success(f"{trend_emoji} **{chart_team}** {trend_direction}: Last 5 games avg = {last_5_avg:.2f} vs Season avg = {season_avg:.2f} ({trend_pct:+.1f}%)")
            else:
                st.info(f"➡️ **{chart_team}** stable: Last 5 games avg = {last_5_avg:.2f} vs Season avg = {season_avg:.2f} ({trend_pct:+.1f}%)")
        
        if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns and len(df_games2_analysis) >= 5:
            with trend_cols[1]:
                last_5_avg2 = df_games2_analysis[selected_metric].tail(5).mean()
                season_avg2 = df_games2_analysis[selected_metric].mean()
                trend_diff2 = last_5_avg2 - season_avg2
                trend_pct2 = (trend_diff2 / season_avg2 * 100) if season_avg2 != 0 else 0
                
                # Determine if higher is better (offensive) or lower is better (defensive)
                lower_is_better = 'opponent' in selected_metric.lower() or 'allowed' in selected_metric.lower()
                
                if abs(trend_pct2) > 5:
                    if lower_is_better:
                        # For defensive stats: lower = improving
                        trend_emoji2 = "📉" if trend_diff2 > 0 else "📈"
                        trend_direction2 = "declining" if trend_diff2 > 0 else "improving"
                    else:
                        # For offensive stats: higher = improving
                        trend_emoji2 = "📈" if trend_diff2 > 0 else "📉"
                        trend_direction2 = "improving" if trend_diff2 > 0 else "declining"
                    st.success(f"{trend_emoji2} **{chart_team2}** {trend_direction2}: Last 5 games avg = {last_5_avg2:.2f} vs Season avg = {season_avg2:.2f} ({trend_pct2:+.1f}%)")
                else:
                    st.info(f"➡️ **{chart_team2}** stable: Last 5 games avg = {last_5_avg2:.2f} vs Season avg = {season_avg2:.2f} ({trend_pct2:+.1f}%)")
        
        st.divider()
    
    # Create chart
    if selected_metric not in df_games_analysis.columns:
        st.error(f"❌ Metric '{selected_metric}' not available in data")
        st.stop()
    
    fig = go.Figure()
    
    # Add Team 1
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df_games_analysis) + 1)),
        y=df_games_analysis[selected_metric],
        mode='lines+markers',
        name=chart_team,
        line=dict(width=2),
        marker=dict(size=8),
        hovertemplate=f"<b>{chart_team}</b><br>Game: %{{x}}<br>{selected_metric_label}: %{{y:.2f}}<extra></extra>"
    ))
    
    # Add Team 2 if available
    if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_games2_analysis) + 1)),
            y=df_games2_analysis[selected_metric],
            mode='lines+markers',
            name=chart_team2,
            line=dict(width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate=f"<b>{chart_team2}</b><br>Game: %{{x}}<br>{selected_metric_label}: %{{y:.2f}}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{selected_metric_label} Over Time ({situation_filter})",
        xaxis_title="Game Number",
        yaxis_title=selected_metric_label,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    summary_cols = st.columns(2 if df_games2_analysis is None else 2)
    
    with summary_cols[0]:
        st.write(f"**{chart_team}**")
        if selected_metric in df_games_analysis.columns:
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{df_games_analysis[selected_metric].mean():.2f}",
                    f"{df_games_analysis[selected_metric].median():.2f}",
                    f"{df_games_analysis[selected_metric].std():.2f}",
                    f"{df_games_analysis[selected_metric].min():.2f}",
                    f"{df_games_analysis[selected_metric].max():.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns:
        with summary_cols[1]:
            st.write(f"**{chart_team2}**")
            stats_df2 = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{df_games2_analysis[selected_metric].mean():.2f}",
                    f"{df_games2_analysis[selected_metric].median():.2f}",
                    f"{df_games2_analysis[selected_metric].std():.2f}",
                    f"{df_games2_analysis[selected_metric].min():.2f}",
                    f"{df_games2_analysis[selected_metric].max():.2f}"
                ]
            })
            st.dataframe(stats_df2, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📊 Data provided by [College Basketball Data API](https://api.collegebasketballdata.com) | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
