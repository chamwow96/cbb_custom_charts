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
    from cbbd.api import games_api, stats_api, teams_api
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
def fetch_team_info(_api_client, team_name):
    """Fetch team info to get conference"""
    try:
        api_instance = teams_api.TeamsApi(_api_client)
        teams = api_instance.get_teams(team=team_name)
        if teams and len(teams) > 0:
            team_info = teams[0].to_dict()
            return team_info.get('conference')
        return None
    except Exception as e:
        return None


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
    
    # Chart type and options
    st.subheader("📈 Chart Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Line Chart", "Rolling Average", "Bar Chart"],
            index=0
        )
    
    with col2:
        # Get team conference info
        team_conference = fetch_team_info(api_client, chart_team)
        
        # Conference comparison toggle
        show_conference_avg = st.checkbox(
            "Compare to Conference Avg",
            value=False,
            help=f"Overlay {team_conference} average on chart" if team_conference else "Conference data not available"
        )
    
    with col3:
        if chart_type == "Rolling Average":
            rolling_window = st.number_input(
                "Rolling Window (games)",
                min_value=2,
                max_value=10,
                value=5,
                step=1
            )
        else:
            st.write("")  # Spacer
    
    with col4:
        st.write("")  # Spacer
    
    # Metric selection
    st.subheader("📊 Select Metric")
    
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
    
    # Fetch conference average data if requested
    conference_avg_series = None
    if show_conference_avg and team_conference:
        with st.spinner(f"Fetching {team_conference} conference data..."):
            try:
                # Fetch all games for the conference
                api_instance = games_api.GamesApi(api_client)
                conf_games = api_instance.get_game_teams(season=season1, conference=team_conference)
                
                # Debug output
                st.write(f"**Debug**: Conference = {team_conference}, Games returned = {len(conf_games) if isinstance(conf_games, list) else 'N/A'}")
                
                if isinstance(conf_games, list) and len(conf_games) > 0:
                    conf_data = []
                    for game in conf_games:
                        try:
                            game_dict = game.to_dict()
                            team_stats = game_dict.get('teamStats', {})
                            opp_stats = game_dict.get('opponentStats', {})
                            
                            # Extract points
                            points_data = team_stats.get('points', {})
                            opp_points_data = opp_stats.get('points', {})
                            fg_data = team_stats.get('fieldGoals', {})
                            opp_fg_data = opp_stats.get('fieldGoals', {})
                            three_pt_data = team_stats.get('threePointers', {})
                            ft_data = team_stats.get('freeThrows', {})
                            reb_data = team_stats.get('rebounds', {})
                            
                            # Helper function to safely extract values
                            def safe_extract(value):
                                if isinstance(value, dict):
                                    return value.get('total', 0)
                                return value if isinstance(value, (int, float)) else 0
                            
                            start_date = game_dict.get('startDate')
                            flat_game = {
                                'team': game_dict.get('team'),
                                'game_date': pd.to_datetime(start_date, errors='coerce') if start_date else pd.NaT,
                                'points': points_data.get('total', 0) if isinstance(points_data, dict) else 0,
                                'opponent_points': opp_points_data.get('total', 0) if isinstance(opp_points_data, dict) else 0,
                                'field_goal_pct': fg_data.get('pct', 0) if isinstance(fg_data, dict) else 0,
                                'opponent_field_goal_pct': opp_fg_data.get('pct', 0) if isinstance(opp_fg_data, dict) else 0,
                                'three_point_pct': three_pt_data.get('pct', 0) if isinstance(three_pt_data, dict) else 0,
                                'free_throw_pct': ft_data.get('pct', 0) if isinstance(ft_data, dict) else 0,
                                'rebounds': reb_data.get('total', 0) if isinstance(reb_data, dict) else 0,
                                'assists': safe_extract(team_stats.get('assists', 0)),
                                'turnovers': safe_extract(team_stats.get('turnovers', 0)),
                                'steals': safe_extract(team_stats.get('steals', 0)),
                                'blocks': safe_extract(team_stats.get('blocks', 0)),
                                'pace': game_dict.get('pace', 0)
                            }
                            conf_data.append(flat_game)
                        except Exception as e:
                            st.write(f"**Debug**: Error processing game: {str(e)}")
                            continue
                    
                    df_conf = pd.DataFrame(conf_data)
                    st.write(f"**Debug**: Conference DF created with {len(df_conf)} rows")
                    st.write(f"**Debug**: Conference DF columns: {list(df_conf.columns)}")
                    st.write(f"**Debug**: Selected metric: {selected_metric}, Available in conf data: {selected_metric in df_conf.columns}")
                    
                    df_conf = df_conf.dropna(subset=['game_date'])
                    df_conf['point_diff'] = df_conf['points'] - df_conf['opponent_points']
                    df_conf = df_conf.sort_values('game_date')
                    
                    # Calculate conference average at each team game date
                    if selected_metric in df_conf.columns:
                        conference_avg_values = []
                        for team_date in df_games_analysis['game_date']:
                            conf_up_to_date = df_conf[df_conf['game_date'] <= team_date]
                            if len(conf_up_to_date) > 0:
                                avg = conf_up_to_date[selected_metric].mean()
                                conference_avg_values.append(avg)
                            else:
                                conference_avg_values.append(None)
                        
                        conference_avg_series = pd.Series(conference_avg_values, index=df_games_analysis.index)
                        overall_conf_avg = df_conf[selected_metric].mean()
                        st.success(f"✓ Loaded {len(df_conf)} conference games (Avg: {overall_conf_avg:.2f})")
                        st.write(f"**Debug**: Conference avg series length: {len(conference_avg_series)}, null count: {conference_avg_series.isna().sum()}")
                    else:
                        st.warning(f"⚠️ Metric '{selected_metric}' not available in conference data. Available: {list(df_conf.columns)}")
                else:
                    st.warning(f"No conference data available for {team_conference}")
            except Exception as e:
                st.error(f"Could not fetch conference data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
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
    
    if chart_type == "Line Chart":
        # Standard line chart
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_games_analysis) + 1)),
            y=df_games_analysis[selected_metric],
            mode='lines+markers',
            name=chart_team,
            line=dict(width=2, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate=f"<b>{chart_team}</b><br>Game: %{{x}}<br>{selected_metric_label}: %{{y:.2f}}<extra></extra>"
        ))
        
        if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games2_analysis) + 1)),
                y=df_games2_analysis[selected_metric],
                mode='lines+markers',
                name=chart_team2,
                line=dict(width=2, dash='dash', color='#ff7f0e'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate=f"<b>{chart_team2}</b><br>Game: %{{x}}<br>{selected_metric_label}: %{{y:.2f}}<extra></extra>"
            ))
        
        # Add conference average
        if conference_avg_series is not None:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games_analysis) + 1)),
                y=conference_avg_series,
                mode='lines',
                name=f'{team_conference} Avg',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ))
    
    elif chart_type == "Rolling Average":
        # Calculate rolling averages
        df_games_analysis[f'{selected_metric}_rolling'] = df_games_analysis[selected_metric].rolling(
            window=rolling_window,
            min_periods=1
        ).mean()
        
        # Plot actual values as markers
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_games_analysis) + 1)),
            y=df_games_analysis[selected_metric],
            mode='markers',
            name=f'{chart_team} Actual',
            marker=dict(size=6, color='lightblue', opacity=0.5)
        ))
        
        # Plot rolling average
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df_games_analysis) + 1)),
            y=df_games_analysis[f'{selected_metric}_rolling'],
            mode='lines',
            name=f'{chart_team} {rolling_window}-Game Avg',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add second team if provided
        if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns:
            df_games2_analysis[f'{selected_metric}_rolling'] = df_games2_analysis[selected_metric].rolling(
                window=rolling_window,
                min_periods=1
            ).mean()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games2_analysis) + 1)),
                y=df_games2_analysis[selected_metric],
                mode='markers',
                name=f'{chart_team2} Actual',
                marker=dict(size=6, color='#ffcc99', opacity=0.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games2_analysis) + 1)),
                y=df_games2_analysis[f'{selected_metric}_rolling'],
                mode='lines',
                name=f'{chart_team2} {rolling_window}-Game Avg',
                line=dict(color='#ff7f0e', width=3)
            ))
        
        # Add conference average
        if conference_avg_series is not None:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games_analysis) + 1)),
                y=conference_avg_series,
                mode='lines',
                name=f'{team_conference} Avg',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ))
    
    elif chart_type == "Bar Chart":
        # Bar chart
        fig.add_trace(go.Bar(
            x=list(range(1, len(df_games_analysis) + 1)),
            y=df_games_analysis[selected_metric],
            name=chart_team,
            marker=dict(color='#2ca02c'),
            opacity=0.8
        ))
        
        if df_games2_analysis is not None and selected_metric in df_games2_analysis.columns:
            fig.add_trace(go.Bar(
                x=list(range(1, len(df_games2_analysis) + 1)),
                y=df_games2_analysis[selected_metric],
                name=chart_team2,
                marker=dict(color='#d62728'),
                opacity=0.8
            ))
        
        # Add conference average
        if conference_avg_series is not None:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_games_analysis) + 1)),
                y=conference_avg_series,
                mode='lines',
                name=f'{team_conference} Avg',
                line=dict(color='red', width=3, dash='dash'),
                opacity=0.8
            ))
    
    # Update layout
    title_suffix = f" vs {team_conference} Avg" if conference_avg_series is not None else ""
    fig.update_layout(
        title=f"{selected_metric_label} Over Time ({situation_filter}){title_suffix}",
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
