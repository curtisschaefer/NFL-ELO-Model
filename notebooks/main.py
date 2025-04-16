#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import date
from datetime import timedelta
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde
from scipy.stats import norm
import statistics
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import re
from datetime import datetime
import pytz
from sklearn.model_selection import TimeSeriesSplit
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import glob

# %%
# Email account credentials
sender_email = "curtis.h.schaefer@gmail.com"
sender_password = "hpxm nhaw jwif waap"
receiver_email = "chschaefer@wisc.edu"

# Create the email content
subject = "ELO Code Notification"
body = "Your Python code has finished running!"

msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

# Function to send email
def send_email():
    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use Gmail's SMTP server
        server.starttls()  # Enable encryption
        server.login(sender_email, sender_password)  # Log in to the server
        server.send_message(msg)  # Send the email
        print("Email sent successfully!")
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

# Simulate long-running code
print("Running code...")  # Replace this with your actual code

# Send the email when the code finishes
send_email()


# %%

# Step 1: Load the dataset
cdf = pd.read_csv('nfl_outcomes.csv')

# Step 2: Convert 'game_date' column to datetime format
cdf['game_date'] = pd.to_datetime(cdf['game_date'])

# Step 3: Compute margin of victory
cdf['margin'] = cdf['total_home_score'] - cdf['total_away_score']

cdf.rename(columns={'home_team': 'home', 'away_team': 'vis'}, inplace=True)

# Step 6: Sort the DataFrame by year and week to maintain chronological order
cdf = cdf.sort_values(by=['year', 'week']).reset_index(drop=True)
cdf = cdf.drop(columns=["game_id"])


# Step 8: Update the Elo dictionary with all teams
teams = pd.concat([cdf['home'], cdf['vis']]).unique()
elo_dict = {team: 1500 for team in teams}

# ✅ Check the updated DataFrame
cdf.tail(30)  # Display the last 30 rows to confirm the Wild Card games are added


# %%
# Function to generate training-validation folds
def generate_time_periods(max_folds=8):
    time_periods_dict = {}

    for num_folds in range(5, max_folds + 1):  # 3 to max_folds (default 8)
        train_periods = []
        total_train_years = 2019  # Training should go up to 2019 only
        val_size = 3  # Validation period is always 4 years

        for i in range(num_folds):
            train_start = 2000 + i  # Training expands year by year
            train_end = total_train_years - (num_folds - (i + 1)) - val_size
            val_start = train_end + 1
            val_end = val_start + val_size - 1

            if val_end <= 2019:  # Ensure validation stays within range
                train_periods.append((train_start, train_end, val_start, val_end))

        time_periods_dict[num_folds] = train_periods  # Store for each fold size

    return time_periods_dict

# === Elo Computation with Team & QB Elo Updates ===
def compute_MSE_and_update_elo(params, cdf, initial_elo, initial_qb_elo):
    """
    Compute MSE and update Elo ratings for teams and QBs dynamically while optimizing parameters.
    
    Parameters:
    - params: List of parameters to optimize (k_one, sigma_one, hfa, team_mean_revert, qb_mean_revert, k_qb).
    - cdf: DataFrame containing game data.
    - initial_elo: Dictionary of initial Elo ratings for each team.
    - initial_qb_elo: Dictionary of initial QB Elo ratings.
    
    Returns:
    - MSE: Mean squared error of predicted vs. actual game margins.
    - Updated team Elo dictionary.
    - Updated QB Elo dictionary.
    """
    # Unpack parameters
    k_one, sigma_one, hfa, team_mean_revert, qb_mean_revert, k_qb = params
    
    # Copy initial Elo ratings
    elo_dict = initial_elo.copy()
    qb_elo_dict = initial_qb_elo.copy()
    
    # Initialize MSE
    mse = 0
    
    # Track the current year for applying mean reversion
    current_year = cdf.iloc[0]['year']
    
    for _, row in cdf.iterrows():
        home = row['home']
        vis = row['vis']
        home_qb = row['home_qb']
        away_qb = row['away_qb']
        margin = row['margin']
        home_qb_epa = row['home_qb_epa']
        away_qb_epa = row['away_qb_epa']
        year = row['year']
        
        # Apply mean reversion at the start of a new year
        # Apply mean reversion at the start of a new year
        if year != current_year:
            current_year = year
    
            # Apply mean reversion for teams
            for team in elo_dict:
                elo_dict[team] -= team_mean_revert * (elo_dict[team] - 1500)
    
    # Apply mean reversion for QBs (only for those who played in the past but not this year)
            for qb in list(qb_elo_dict.keys()):  
                qb_elo_dict[qb] -= qb_mean_revert * (qb_elo_dict[qb] - 1500)  # Gradual decay to 1500

        
        # Get current team and QB Elo ratings
        elo_home = elo_dict[home]
        elo_vis = elo_dict[vis]
        qb_elo_home = qb_elo_dict.get(home_qb, 1400)
        qb_elo_vis = qb_elo_dict.get(away_qb, 1400)
        
        # **Make prediction using current QB Elo (before updating)**
        elo_home_adj = elo_home + (qb_elo_home - 1500)
        elo_vis_adj = elo_vis + (qb_elo_vis - 1500)
        predicted_margin = abs(elo_home_adj + hfa - elo_vis_adj) / sigma_one
        #predicted_margin = (abs(elo_home_adj + hfa - elo_vis_adj) + (abs(elo_home_adj + hfa - elo_vis_adj) ** sigma_one)) / 100
        if elo_vis_adj > elo_home_adj + hfa:
            predicted_margin *= -1
        
        # Compute squared error for MSE calculation
        mse += (margin - predicted_margin) ** 2
        
        # **Update Team Elo ratings**
        update = k_one * (margin - predicted_margin)
        elo_dict[home] += update
        elo_dict[vis] -= update
        
        # **Compute QB performance deviation AFTER the prediction**
        expected_qb_epa_home = (qb_elo_home - 1500) / 1000 - 0.05
        expected_qb_epa_vis = (qb_elo_vis - 1500) / 1000 - 0.05

        qb_performance_delta_home = home_qb_epa - expected_qb_epa_home
        qb_performance_delta_vis = away_qb_epa - expected_qb_epa_vis

        # **Now update QB Elo ratings**
        qb_elo_home += k_qb * qb_performance_delta_home
        qb_elo_vis += k_qb * qb_performance_delta_vis
        
        # Store updated QB Elo
        qb_elo_dict[home_qb] = qb_elo_home
        qb_elo_dict[away_qb] = qb_elo_vis
    
    return mse, elo_dict, qb_elo_dict  # Return MSE, team Elo, and QB Elo

# === Cross-Validation Function ===
def cross_validation_objective(params, cdf, time_periods):
    """
    Objective function for cross-validation.
    Computes the average MSE across all time periods (folds).
    """
    total_mse = 0

    for train_start, train_end, test_start, test_end in time_periods:
        # Split dataset into training and testing periods
        train_df = cdf[(cdf['year'] >= train_start) & (cdf['year'] <= train_end)]
        test_df = cdf[(cdf['year'] >= test_start) & (cdf['year'] <= test_end)]
        
        # Initialize Elo ratings to 1500
        initial_elo = {team: 1500 for team in pd.concat([train_df['vis'], train_df['home']]).unique()}
        initial_qb_elo = {qb: 1400 for qb in pd.concat([train_df['home_qb'], train_df['away_qb']]).unique()}

        # Train the Elo function
        _, final_elo_dict, final_qb_elo_dict = compute_MSE_and_update_elo(params, train_df, initial_elo, initial_qb_elo)
        
        # Evaluate on the test set
        mse, _, _ = compute_MSE_and_update_elo(params, test_df, final_elo_dict, final_qb_elo_dict)
        total_mse += mse

    return total_mse / len(time_periods)



# Generate training-validation folds
time_periods_dict = generate_time_periods(max_folds=5)

# Define the final test set (2021-2024)
test_df = cdf[(cdf['year'] >= 2020) & (cdf['year'] <= 2024)]

# Store results for different fold numbers
results = {}

for num_folds, time_periods in time_periods_dict.items():
    print(f"\n=== Running {num_folds}-Fold Cross-Validation ===")

    # Optimize K, Sigma, HFA, Team Mean Revert, QB Mean Revert, K_QB using L-BFGS-B
    initial_params = [2.28, 50, 109, 0.47, 0.1, 100]  # Initial values
    bounds = [(0, 20), (0, 100), (0, 200), (0, 1), (0, 1), (0, 200)]  # Parameter bounds

    result = minimize(
        lambda params: cross_validation_objective(params, cdf, time_periods),
        x0=initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Extract optimized parameters
    best_k, best_sigma, best_hfa, best_team_mean_revert, best_qb_mean_revert, best_k_qb = result.x

    # Train Elo using the best params on all pre-2021 data
    full_train_df = cdf[(cdf['year'] < 2020)]
    initial_elo = {team: 1500 for team in pd.concat([full_train_df['vis'], full_train_df['home']]).unique()}
    initial_qb_elo = {qb: 1400 for qb in pd.concat([full_train_df['home_qb'], full_train_df['away_qb']]).unique()}

    _, final_elo_dict, final_qb_elo_dict = compute_MSE_and_update_elo(
        [best_k, best_sigma, best_hfa, best_team_mean_revert, best_qb_mean_revert, best_k_qb],
        full_train_df, initial_elo, initial_qb_elo
    )

    # Evaluate the best model on the test set (2021-2024)
    test_mse, final_elo_dict, final_qb_elo_dict = compute_MSE_and_update_elo(
        [best_k, best_sigma, best_hfa, best_team_mean_revert, best_qb_mean_revert, best_k_qb],
        test_df, final_elo_dict, final_qb_elo_dict
    )

    # Store results
    results[num_folds] = {
        'MSE': test_mse,
        'params': (best_k, best_sigma, best_hfa, best_team_mean_revert, best_qb_mean_revert, best_k_qb),
        'team_elo': final_elo_dict,
        'qb_elo': final_qb_elo_dict
    }

    print(f"Optimized K: {best_k:.4f}, Sigma: {best_sigma:.4f}, HFA: {best_hfa:.4f}, "
          f"Team Mean Revert: {best_team_mean_revert:.4f}, QB Mean Revert: {best_qb_mean_revert:.4f}, K_QB: {best_k_qb:.4f}")
    print(f"Test Set MSE (2020-2024): {test_mse:.4f}")

# Find the best model based on test MSE
best_folds = min(results, key=lambda x: results[x]['MSE'])
best_params = results[best_folds]['params']
best_team_elo = results[best_folds]['team_elo']
best_qb_elo = results[best_folds]['qb_elo']

print(f"\nBest Fold Count: {best_folds}")
print(f"Best Parameters: K={best_params[0]}, Sigma={best_params[1]}, HFA={best_params[2]}, "
      f"Team Mean Revert={best_params[3]}, QB Mean Revert={best_params[4]}, K_QB={best_params[5]}")
print(f"Best Test MSE (2020-2024): {results[best_folds]['MSE']:.4f}")

# Print final team Elo ratings (sorted) - NOW REPRESENTING THE MOST RECENT ELO
sorted_team_elo = sorted(best_team_elo.items(), key=lambda x: x[1], reverse=True)
print("\nFinal Team Elo Ratings (Top 10) - Current Elo:")
for team, elo in sorted_team_elo[:10]:
    print(f"{team}: {elo:.2f}")

# Print final QB Elo ratings (sorted) - NOW REPRESENTING THE MOST RECENT ELO
sorted_qb_elo = sorted(best_qb_elo.items(), key=lambda x: x[1], reverse=True)
print("\nFinal QB Elo Ratings (Top 10) - Current Elo:")
for qb, elo in sorted_qb_elo[:10]:
    print(f"{qb}: {elo:.2f}")


# %%
best_team_elo_sorted = dict(sorted(best_team_elo.items(), key=lambda x: x[1], reverse=True))
best_team_elo_sorted

# %%
# Get QBs who played in 2024 (FAST using .loc)
qbs_2024 = set(cdf.loc[cdf['year'] == 2024, 'home_qb']).union(set(cdf.loc[cdf['year'] == 2024, 'away_qb']))

# Filter best_qb_elo to keep only QBs who played in 2024
best_qb_elo_filtered = {qb: elo for qb, elo in best_qb_elo.items() if qb in qbs_2024}
best_qb_elo_sorted = dict(sorted(best_qb_elo_filtered.items(), key=lambda x: x[1], reverse=True))
best_qb_elo_sorted

# %%
# Get current team and QB Elo ratings
elo_home = best_team_elo_sorted["PHI"]
elo_vis = best_team_elo_sorted["KC"]
qb_elo_home = best_qb_elo_sorted['J.Hurts']
qb_elo_vis = best_qb_elo_sorted['P.Mahomes']
sigma = best_params[1]
hfa = 0
elo_home_adj = elo_home + (qb_elo_home - 1500)
elo_vis_adj = elo_vis + (qb_elo_vis - 1500)
predicted_margin = (elo_home_adj + hfa - elo_vis_adj) / sigma
predicted_margin

# %%
elo_home = best_team_elo['KC']
elo_vis = best_team_elo['BUF']
qb_elo_home = best_qb_elo['P.Mahomes']
qb_elo_vis = best_qb_elo['J.Allen']
elo_home_adj = elo_home + (qb_elo_home - 1500)
elo_vis_adj = elo_vis + (qb_elo_vis - 1500)

hfa = best_params[1]
predicted_margin = abs(elo_home_adj + hfa - elo_vis_adj) / 50
if elo_vis_adj > elo_home_adj + hfa:
    predicted_margin *= -1
predicted_margin

# %%
elo_home = best_team_elo['PHI']
elo_vis = best_team_elo['WAS']
qb_elo_home = best_qb_elo['J.Hurts']
qb_elo_vis = best_qb_elo['J.Daniels']
elo_home_adj = elo_home + (qb_elo_home - 1500)
elo_vis_adj = elo_vis + (qb_elo_vis - 1500)
hfa = best_params[1]
predicted_margin = abs(elo_home_adj + hfa - elo_vis_adj) / 50
if elo_vis_adj > elo_home_adj + hfa:
    predicted_margin *= -1
predicted_margin

# %%
# Define the correct file path pattern
file_path = "../ELO/Odds/nfl_odds_*.csv"  # Update with your actual path

# Load all CSV files
all_files = glob.glob(file_path)
all_files.sort()

# Initialize an empty list to store data
data_list = []

# Process each CSV file and load into a DataFrame
for file in all_files:
    try:
        df = pd.read_csv(file)
        df["file_name"] = os.path.basename(file)  # Keep track of source file
        #print(f"Loaded: {os.path.basename(file)} | Shape: {df.shape}")  # Debugging info
        data_list.append(df)

    except Exception as e:
        print(f"Error loading {os.path.basename(file)}: {e}")

# Combine all loaded files into a single DataFrame
if not data_list:
    raise ValueError("No valid data was loaded. Check your CSV files.")

odds_df = pd.concat(data_list, ignore_index=True)



# %%



# Step 1: Ensure Correct Team Name Mapping
teamcodes = {'Chicago Bears':'CHI', 'Philadelphia Eagles':'PHI', 'Los Angeles Rams':'LA',
             'Tampa Bay Buccaneers':'TB', 'Carolina Panthers':'CAR', 'Buffalo Bills':'BUF',
             'Tennessee Titans':'TEN', 'Seattle Seahawks':'SEA', 'New York Jets':'NYJ',
             'San Francisco 49ers':'SF', 'Detroit Lions':'DET', 'Jacksonville Jaguars':'JAX',
             'Cleveland Browns':'CLE', 'Pittsburgh Steelers':'PIT', 'Cincinnati Bengals':'CIN',
             'Los Angeles Chargers':'LAC', 'Atlanta Falcons':'ATL', 'Minnesota Vikings':'MIN',
             'Arizona Cardinals':'ARI', 'Las Vegas Raiders':'LV', 'Kansas City Chiefs':'KC',
             'Miami Dolphins':'MIA', 'Green Bay Packers':'GB', 'New York Giants':'NYG',
             'Dallas Cowboys':'DAL', 'New Orleans Saints':'NO', 'Indianapolis Colts':'IND',
             'Baltimore Ravens':'BAL', 'Washington Football Team':'WAS', 'Washington Commanders':'WAS',
             'New England Patriots':'NE', 'Houston Texans':'HOU', 'Denver Broncos':'DEN'}

# Step 2: Filter for only spread bets
spread_df = odds_df[odds_df["market_key"] == "spreads"].copy()

# Step 3: Convert timestamps to datetime and adjust to **Eastern Time**
spread_df["commence_time"] = pd.to_datetime(spread_df["commence_time"]).dt.tz_convert('US/Eastern')
spread_df["market_last_update"] = pd.to_datetime(spread_df["market_last_update"]).dt.tz_convert('US/Eastern')

# Step 4: Adjust `year` for January & February games (count as previous season)
spread_df["year"] = spread_df["commence_time"].dt.year
spread_df.loc[spread_df["commence_time"].dt.month <= 2, "year"] -= 1

# Step 5: Extract unique **real games** from `cdf` with actual dates
cdf_games = cdf[cdf["year"] >= 2020][["home", "vis", "year", "week", "game_date"]].drop_duplicates()
cdf_games["game_date"] = pd.to_datetime(cdf_games["game_date"]).dt.tz_localize('US/Eastern')  # Ensure timezone consistency
cdf_games.rename(columns={"home": "home_team", "vis": "away_team"}, inplace=True)

# Apply team name mapping
spread_df["home_team"] = spread_df["home_team"].map(teamcodes)
spread_df["away_team"] = spread_df["away_team"].map(teamcodes)
spread_df["outcome_name"] = spread_df["outcome_name"].map(teamcodes)

# Merge with `cdf_games` to get correct weeks and dates
spread_df = spread_df.merge(cdf_games, on=["home_team", "away_team", "year"], how="left")

# Step 6: **Remove Invalid January/February Odds**
# Games where odds were listed in January/February but **no game was played within 14 days**
spread_df["days_until_game"] = (spread_df["game_date"] - spread_df["market_last_update"]).dt.days

# Remove any game where odds were listed in January/February **and** no game is within 14 days
spread_df = spread_df[~((spread_df["commence_time"].dt.month <= 2) & (spread_df["days_until_game"].abs() > 14))]

# Step 7: Identify Favorite & Assign Correct Spread Signs
spread_df["home_favored"] = ((spread_df["outcome_name"] == spread_df["home_team"]) & (spread_df["outcome_point"] < 0)) | \
                            ((spread_df["outcome_name"] == spread_df["away_team"]) & (spread_df["outcome_point"] > 0))

# If home is NOT favored, then away must be favored
spread_df["away_favored"] = ~spread_df["home_favored"]

# Assign Correct Spread Signs
spread_df["adjusted_spread"] = np.where(
    spread_df["home_favored"], abs(spread_df["outcome_point"]),  # Home team favored → Keep positive
    -abs(spread_df["outcome_point"])  # Away team favored → Make negative
)

# ✅ **Debugging Prints**
print("✅ Total unique games in cdf_games:", len(set(cdf_games.apply(lambda row: (row["home_team"], row["away_team"], row["year"], row["week"]), axis=1))))
print("✅ Total rows in spread_df AFTER filtering:", len(spread_df))

spread_df.head()


# %%
# Ensure correct data types
spread_df["week"] = spread_df["week"].astype(float)
spread_df["commence_time"] = pd.to_datetime(spread_df["commence_time"]).dt.tz_convert('US/Eastern')
spread_df["market_last_update"] = pd.to_datetime(spread_df["market_last_update"]).dt.tz_convert('US/Eastern')

# Step 1: Find the last game of **Week 1** (latest commence_time)
week_1_end_dates = spread_df[spread_df["week"] == 1].groupby("year")["commence_time"].max().reset_index()
week_1_end_dates.rename(columns={"commence_time": "week_1_last_game"}, inplace=True)

# Merge this back into spread_df to align the correct "last game date" for each season
spread_df = spread_df.merge(week_1_end_dates, on="year", how="left")

# ✅ Ensure `week_1_last_game` is in the same timezone as `market_last_update`
spread_df["week_1_last_game"] = spread_df["week_1_last_game"].dt.tz_convert('US/Eastern')

# Step 2: **Filter out non-Week 1 games if their market update is before the last game of Week 1**
spread_df = spread_df[~(
    (spread_df["week"] > 1) & 
    (spread_df["market_last_update"] < spread_df["week_1_last_game"])
)]

# Step 3: Drop the unnecessary column
spread_df = spread_df.drop(columns=["week_1_last_game"])

# ✅ Debugging prints
print(f"✅ After filtering: {len(spread_df)} rows remain.")
spread_df.head()



# %%
# Step 1: Keep only relevant columns before merging
filtered_spread_df = spread_df[[
    "year", "week", "home_team", "away_team", 
    "adjusted_spread", "outcome_price", "bookmaker_key", "market_last_update"
]]

# Step 2: Filter for valid odds (-125 to +125)
filtered_spread_df = filtered_spread_df[
    (filtered_spread_df["outcome_price"] == -110)
]

# Step 3: Identify Opening and Closing Spreads
opening_spreads = (
    filtered_spread_df
    .sort_values(by=["year", "week", "home_team", "away_team", "bookmaker_key", "market_last_update"])
    .groupby(["year", "week", "home_team", "away_team", "bookmaker_key"])
    .first()
    .reset_index()
)

closing_spreads = (
    filtered_spread_df
    .sort_values(by=["year", "week", "home_team", "away_team", "bookmaker_key", "market_last_update"])
    .groupby(["year", "week", "home_team", "away_team", "bookmaker_key"])
    .last()
    .reset_index()
)

# Step 4: Get lowest & highest opening spreads per game within odds range
lowest_opening_spread = opening_spreads.loc[
    opening_spreads.groupby(["year", "week", "home_team", "away_team"])["adjusted_spread"].idxmin()
].rename(columns={
    "bookmaker_key": "lowest_opening_book", 
    "adjusted_spread": "lowest_opening_spread",
    "outcome_price": "lowest_opening_odds"
})

highest_opening_spread = opening_spreads.loc[
    opening_spreads.groupby(["year", "week", "home_team", "away_team"])["adjusted_spread"].idxmax()
].rename(columns={
    "bookmaker_key": "highest_opening_book", 
    "adjusted_spread": "highest_opening_spread",
    "outcome_price": "highest_opening_odds"
})

# Step 5: Get lowest & highest closing spreads per game within odds range
lowest_closing_spread = closing_spreads.loc[
    closing_spreads.groupby(["year", "week", "home_team", "away_team"])["adjusted_spread"].idxmin()
].rename(columns={
    "bookmaker_key": "lowest_closing_book", 
    "adjusted_spread": "lowest_closing_spread",
    "outcome_price": "lowest_closing_odds"
})

highest_closing_spread = closing_spreads.loc[
    closing_spreads.groupby(["year", "week", "home_team", "away_team"])["adjusted_spread"].idxmax()
].rename(columns={
    "bookmaker_key": "highest_closing_book", 
    "adjusted_spread": "highest_closing_spread",
    "outcome_price": "highest_closing_odds"
})

# Step 6: Merge all into a single DataFrame
spread_summary_df = (
    lowest_opening_spread[["year", "week", "home_team", "away_team", "lowest_opening_spread", "lowest_opening_odds", "lowest_opening_book"]]
    .merge(highest_opening_spread[["year", "week", "home_team", "away_team", "highest_opening_spread", "highest_opening_odds", "highest_opening_book"]],
           on=["year", "week", "home_team", "away_team"], how="inner")
    .merge(lowest_closing_spread[["year", "week", "home_team", "away_team", "lowest_closing_spread", "lowest_closing_odds", "lowest_closing_book"]],
           on=["year", "week", "home_team", "away_team"], how="inner")
    .merge(highest_closing_spread[["year", "week", "home_team", "away_team", "highest_closing_spread", "highest_closing_odds", "highest_closing_book"]],
           on=["year", "week", "home_team", "away_team"], how="inner")
)

spread_summary_df.tail(20)



# %%
# Initialize Elo ratings (all teams start at 1500, QBs at 1400)
initial_elo = {team: 1500 for team in pd.concat([cdf['home'], cdf['vis']]).unique()}
initial_qb_elo = {qb: 1400 for qb in pd.concat([cdf['home_qb'], cdf['away_qb']]).unique()}

# Copy initial Elo ratings to track updates
elo_dict = initial_elo.copy()
qb_elo_dict = initial_qb_elo.copy()

# Store game-by-game Elo predictions
elo_predictions = []

# Track current year for mean reversion
current_year = cdf.iloc[0]['year']

# Loop through each game in the dataset
for _, row in cdf.iterrows():
    home = row['home']
    vis = row['vis']
    home_qb = row['home_qb']
    away_qb = row['away_qb']
    margin = row['margin']
    home_qb_epa = row['home_qb_epa']
    away_qb_epa = row['away_qb_epa']
    year = row['year']

    # Apply mean reversion at the start of a new year
    if year != current_year:
        current_year = year

        # Apply mean reversion for teams
        for team in elo_dict:
            elo_dict[team] -= best_team_mean_revert * (elo_dict[team] - 1500)

        # Apply mean reversion for QBs who didn't play this year
        current_year_qbs = set(cdf.loc[cdf['year'] == year, 'home_qb']).union(set(cdf.loc[cdf['year'] == year, 'away_qb']))
        inactive_qbs = set(qb_elo_dict.keys()) - current_year_qbs  # Fast set difference

        for qb in inactive_qbs:
            qb_elo_dict[qb] -= best_qb_mean_revert * (qb_elo_dict[qb] - 1500)

    # Get current Elo ratings for teams and QBs
    elo_home = elo_dict[home]
    elo_vis = elo_dict[vis]
    qb_elo_home = qb_elo_dict.get(home_qb, 1400)
    qb_elo_vis = qb_elo_dict.get(away_qb, 1400)

    # Adjusted team Elo with QB influence
    elo_home_adj = elo_home + (qb_elo_home - 1500)
    elo_vis_adj = elo_vis + (qb_elo_vis - 1500)

    # Compute predicted margin (Elo spread)
    elo_spread = abs(elo_home_adj + best_hfa - elo_vis_adj) / best_sigma
    if elo_vis_adj > elo_home_adj + best_hfa:
        elo_spread *= -1

    # Store prediction data
    elo_predictions.append({
        "year": year, "week": row["week"], "home_team": home, "away_team": vis,
        "home_qb": home_qb, "away_qb": away_qb, "elo_spread": elo_spread
    })

    # Update Team Elo ratings
    update = best_k * (margin - elo_spread)
    elo_dict[home] += update
    elo_dict[vis] -= update

    # Compute QB performance deviation
    expected_qb_epa_home = (qb_elo_home - 1500) / 1000 - 0.05
    expected_qb_epa_vis = (qb_elo_vis - 1500) / 1000 - 0.05
    qb_performance_delta_home = home_qb_epa - expected_qb_epa_home
    qb_performance_delta_vis = away_qb_epa - expected_qb_epa_vis

    # Update QB Elo ratings
    qb_elo_home += best_k_qb * qb_performance_delta_home
    qb_elo_vis += best_k_qb * qb_performance_delta_vis

    # Store updated QB Elo
    qb_elo_dict[home_qb] = qb_elo_home
    qb_elo_dict[away_qb] = qb_elo_vis

# Convert list to DataFrame
elo_predictions_df = pd.DataFrame(elo_predictions)



# %%
elo_predictions_df

# %%
# Extract actual game results from cdf
actual_results = cdf[["year", "week", "home", "vis", "margin"]].rename(
    columns={"home": "home_team", "vis": "away_team", "margin": "actual_margin"}
)


# %%
betting_df = spread_summary_df.merge(elo_predictions_df, on=["year", "week", "home_team", "away_team"], how="inner")
betting_df = betting_df.merge(actual_results, on=["year", "week", "home_team", "away_team"], how="inner")
betting_df.tail(38)

# %%
betting_test = betting_df.copy()
betting_test.drop(columns=["lowest_closing_spread", 'lowest_closing_odds', 'lowest_closing_book', 'highest_closing_spread', 'highest_closing_odds', 'highest_closing_book'], inplace=True)  # Modifies only the copy
betting_test.head(50)


# %%
betting_test.to_csv('betting_df.csv', index=False)


# %%
def determine_bet_team(elo_spread, 
                       opening_spread_low, opening_spread_high, 
                       closing_spread_low, closing_spread_high, 
                       opening_odds_low, opening_odds_high, 
                       closing_odds_low, closing_odds_high,
                       home_team, away_team, threshold):
    """
    Determines the team to bet on and the odds based on Elo spread differences.
    
    Parameters:
    - elo_spread: Expected spread from Elo model.
    - opening_spread_low / opening_spread_high: Lowest and highest opening spreads.
    - closing_spread_low / closing_spread_high: Lowest and highest closing spreads.
    - opening_odds_low / opening_odds_high: Odds associated with the opening spreads.
    - closing_odds_low / closing_odds_high: Odds associated with the closing spreads.
    - home_team: Home team name.
    - away_team: Away team name.
    - threshold: Minimum absolute difference required to place a bet.
    
    Returns:
    - bet_team_opening: Team to bet on at opening line.
    - spread_used_opening: The spread used for the bet.
    - odds_used_opening: The odds corresponding to the chosen spread.
    - bet_team_closing: Team to bet on at closing line.
    - spread_used_closing: The spread used for the bet.
    - odds_used_closing: The odds corresponding to the chosen spread.
    """

    # Select the opening spread with the greater absolute difference
    if abs(elo_spread - opening_spread_low) > abs(elo_spread - opening_spread_high):
        spread_used_opening = opening_spread_low
        odds_used_opening = opening_odds_low
    else:
        spread_used_opening = opening_spread_high
        odds_used_opening = opening_odds_high
    
    # Select the closing spread with the greater absolute difference
    if abs(elo_spread - closing_spread_low) > abs(elo_spread - closing_spread_high):
        spread_used_closing = closing_spread_low
        odds_used_closing = closing_odds_low
    else:
        spread_used_closing = closing_spread_high
        odds_used_closing = closing_odds_high

    # Determine the team to bet on at opening
    if abs(elo_spread - spread_used_opening) > threshold:
        bet_team_opening = away_team if elo_spread < spread_used_opening else home_team
    else:
        bet_team_opening = "No Bet"

    # Determine the team to bet on at closing
    if abs(elo_spread - spread_used_closing) > threshold:
        bet_team_closing = away_team if elo_spread < spread_used_closing else home_team
    else:
        bet_team_closing = "No Bet"

    return bet_team_opening, spread_used_opening, odds_used_opening, bet_team_closing, spread_used_closing, odds_used_closing

# Apply function to determine bet team & odds
threshold = 2.24  # Placeholder, to be optimized later
betting_df[["bet_team_opening", "spread_used_opening", "odds_used_opening", 
            "bet_team_closing", "spread_used_closing", "odds_used_closing"]] = betting_df.apply(
    lambda row: determine_bet_team(row["elo_spread"], 
                                   row["lowest_opening_spread"], row["highest_opening_spread"],
                                   row["lowest_closing_spread"], row["highest_closing_spread"],
                                   row["lowest_opening_odds"], row["highest_opening_odds"],
                                   row["lowest_closing_odds"], row["highest_closing_odds"],
                                   row["home_team"], row["away_team"], threshold), 
    axis=1, result_type="expand"
)

# ✅ Print Sample Output to Verify
print(betting_df[["bet_team_opening", "spread_used_opening", "odds_used_opening", 
                  "bet_team_closing", "spread_used_closing", "odds_used_closing"]].head())


# %%
# Function to determine ATS winner using the correct spread
def determine_ats_winner(actual_margin, spread_used, home_team, away_team):
    if spread_used is None:
        return "No Bet"  # No bet placed, so no ATS winner applies
    if actual_margin > spread_used:
        return home_team
    elif actual_margin < spread_used:
        return away_team
    else:
        return "Push"  # If actual margin exactly equals the spread

# Apply function to check ATS winner based on the spread used for betting
betting_df["ats_winner_opening"] = betting_df.apply(
    lambda row: determine_ats_winner(row["actual_margin"], row["spread_used_opening"], 
                                     row["home_team"], row["away_team"]), axis=1
)

betting_df["ats_winner_closing"] = betting_df.apply(
    lambda row: determine_ats_winner(row["actual_margin"], row["spread_used_closing"], 
                                     row["home_team"], row["away_team"]), axis=1
)


# %%
def calculate_bet_winnings(bet_amount, odds):
    """
    Calculate the net winnings based on the bet amount and American odds.

    Parameters:
    - bet_amount: The amount of money wagered.
    - odds: American odds for the bet.

    Returns:
    - The net profit if the bet wins.
    """
    return np.where(
        odds < 0,  
        (bet_amount * 100) / abs(odds),  # Winnings for negative odds
        (bet_amount * odds) / 100        # Winnings for positive odds
    )

def calculate_profit(row, bet_amount_multiplier=1.0):
    """
    Calculate profit for each bet (opening and closing) based on bet amount and results.

    Parameters:
    - row: A row from the betting DataFrame.
    - bet_amount_multiplier: A scaling factor for bet size to optimize strategy.

    Returns:
    - A tuple containing profit for opening and closing bets, along with their respective bet sizes.
    """
    # Initialize profits and bet sizes
    profit_opening, profit_closing = 0.0, 0.0
    bet_size_opening, bet_size_closing = 0.0, 0.0

    # Handle Opening Bet
    if row["bet_team_opening"] != "No Bet" and row["ats_winner_opening"] != "push":
        bet_size_opening = bet_amount_multiplier * 100  # Base bet size
        bet_won_opening = row["bet_team_opening"] == row["ats_winner_opening"]
        winnings_opening = calculate_bet_winnings(bet_size_opening, row["odds_used_opening"])
        profit_opening = winnings_opening if bet_won_opening else -bet_size_opening

    # Handle Closing Bet
    if row["bet_team_closing"] != "No Bet" and row["ats_winner_closing"] != "push":
        bet_size_closing = bet_amount_multiplier * 100  # Base bet size
        bet_won_closing = row["bet_team_closing"] == row["ats_winner_closing"]
        winnings_closing = calculate_bet_winnings(bet_size_closing, row["odds_used_closing"])
        profit_closing = winnings_closing if bet_won_closing else -bet_size_closing

    return profit_opening, profit_closing, bet_size_opening, bet_size_closing

# Apply the function to the DataFrame for both opening and closing bets
betting_df[["profit_opening", "profit_closing", "bet_size_opening", "bet_size_closing"]] = betting_df.apply(
    lambda row: calculate_profit(row), axis=1, result_type="expand"
)

# ✅ Print Results
betting_df.tail(22)


# %%
# Total amount of money wagered (sum of bet sizes)
total_bets_opening = betting_df["bet_size_opening"].sum()  # Sum actual bet amounts
total_bets_closing = betting_df["bet_size_closing"].sum()  # Sum actual bet amounts

# Total profit
total_profit_opening = betting_df["profit_opening"].sum()
total_profit_closing = betting_df["profit_closing"].sum()

# ROI Calculation (Return on Investment)
roi_opening = (total_profit_opening / total_bets_opening) * 100 if total_bets_opening > 0 else 0
roi_closing = (total_profit_closing / total_bets_closing) * 100 if total_bets_closing > 0 else 0

# Compute total number of games
total_games = len(betting_df)

# Compute percentage of games with bets placed
pct_games_bet_opening = (betting_df["bet_size_opening"] > 0).sum() / total_games * 100 if total_games > 0 else 0
pct_games_bet_closing = (betting_df["bet_size_closing"] > 0).sum() / total_games * 100 if total_games > 0 else 0

# Display final betting results
print(f"Total Games: {total_games}")
print(f"Total Bets on Opening Line: ${total_bets_opening:.2f}, Profit: ${total_profit_opening:.2f}, ROI: {roi_opening:.2f}%, % of Games Bet: {pct_games_bet_opening:.2f}%")
print(f"Total Bets on Closing Line: ${total_bets_closing:.2f}, Profit: ${total_profit_closing:.2f}, ROI: {roi_closing:.2f}%, % of Games Bet: {pct_games_bet_closing:.2f}%")

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

def determine_bet_team(row, threshold):
    """
    Determines the best team to bet on based on Elo spread and available lines.
    """
    diff_low = abs(row["elo_spread"] - row["lowest_opening_spread"])
    diff_high = abs(row["elo_spread"] - row["highest_opening_spread"])

    if diff_low > threshold or diff_high > threshold:
        if diff_low > diff_high:
            return (row["home_team"] if row["elo_spread"] > row["lowest_opening_spread"] else row["away_team"], 
                    row["lowest_opening_spread"], row["lowest_opening_odds"])
        else:
            return (row["home_team"] if row["elo_spread"] > row["highest_opening_spread"] else row["away_team"], 
                    row["highest_opening_spread"], row["highest_opening_odds"])
    
    return ("No Bet", 0.0, -110)  # Default values for no bet

def calculate_profit(row):
    """
    Compute the profit per bet, ensuring that every bet accounts for the vig (-110 odds).
    If it's a push, profit is 0.
    """
    if row["bet_team"] == "No Bet":
        return 0

    bet_amount = row["bet_size"]  # Bet size is now dynamic

    if row["ats_winner"] == "push":
        return 0  # Push = No profit/loss

    bet_won = row["bet_team"] == row["ats_winner"]

    if bet_won:
        return bet_amount * (100 / 110)  # Adjust for -110 odds
    else:
        return -bet_amount  # Lose entire wagered amount

def evaluate_betting_strategy(betting_df, threshold, bet_multiplier):
    """
    Evaluates the betting strategy, computes profit, ROI, and % of games bet.
    """
    betting_df = betting_df.copy()

    # Determine bet team and odds used
    betting_df[["bet_team", "spread_used", "odds_used"]] = betting_df.apply(
        lambda row: determine_bet_team(row, threshold), axis=1, result_type="expand"
    )

    # Assign ATS winner
    betting_df["ats_winner"] = np.where(
        betting_df["actual_margin"] > betting_df["spread_used"], betting_df["home_team"],
        np.where(betting_df["actual_margin"] < betting_df["spread_used"], betting_df["away_team"], "push")
    )

    # Compute bet size dynamically
    betting_df["bet_size"] = betting_df.apply(
        lambda row: bet_multiplier * abs(row["elo_spread"] - row["spread_used"]) 
        if row["bet_team"] != "No Bet" else 0, axis=1
    )

    # Compute profits
    betting_df["profit"] = betting_df.apply(calculate_profit, axis=1)

    # Fix ROI calculation
    total_profit = betting_df["profit"].sum()
    total_wagered = betting_df["bet_size"].sum()  # Total amount of money wagered
    num_bets = (betting_df["bet_team"] != "No Bet").sum()
    pct_games_bet = (num_bets / len(betting_df)) * 100
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0  # Corrected ROI

    return {"profit": total_profit, "pct_games_bet": pct_games_bet, "roi": roi, "betting_df": betting_df}

def loss_function(params, betting_df, lambda_value):
    """
    Compute the risk-penalized profit function (ROI + Variance Control) with bet sizing.
    """
    threshold, bet_multiplier = params
    results = evaluate_betting_strategy(betting_df, threshold, bet_multiplier)

    total_profit = results["profit"]
    betting_df = results["betting_df"]
    variance = betting_df[betting_df.bet_size > 0]['profit'].var()
    
    penalized_loss = -(total_profit / (1 + .1 * variance)) 
    #print(penalized_loss) # Penalizes high variance
    return penalized_loss

def grid_search_betting_strategy(betting_df, threshold_values, bet_multipliers, n_splits=5):
    """
    Performs Grid Search with Cross-Validation to find the best betting threshold and bet size multiplier.
    Uses the loss function to choose the best parameters based on cross-validation performance.
    """
    best_params = None
    best_loss = float("inf")  # Track best (minimized) loss
    best_profit = 0
    best_roi = 0
    best_pct_games_bet = 0
    best_total_games_bet = 0
    results = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for threshold in threshold_values:
        for bet_multiplier in bet_multipliers:
            total_losses = []
            total_profits = []
            total_rois = []
            total_pct_games_bets = []
            total_games_bet = []

            for train_idx, val_idx in kf.split(betting_df):
                train_df, val_df = betting_df.iloc[train_idx], betting_df.iloc[val_idx]

                # Compute loss function on validation set
                loss = loss_function([threshold, bet_multiplier], val_df, lambda_value=0.1)

                # Compute additional metrics for tracking
                results_dict = evaluate_betting_strategy(val_df, threshold, bet_multiplier)

                total_losses.append(loss)
                total_profits.append(results_dict["profit"])
                total_rois.append(results_dict["roi"])
                total_pct_games_bets.append(results_dict["pct_games_bet"])
                total_games_bet.append((results_dict["pct_games_bet"] / 100) * len(val_df))  # Convert % to actual games bet

            avg_loss = np.mean(total_losses)  # Minimize loss
            avg_profit = np.mean(total_profits)
            avg_roi = np.mean(total_rois)
            avg_pct_games_bet = np.mean(total_pct_games_bets)
            avg_games_bet = np.mean(total_games_bet)  # Average total games bet

            results.append((threshold, bet_multiplier, avg_profit, avg_roi, avg_pct_games_bet, avg_games_bet, avg_loss))

            # Select the best parameters based on lowest cross-validation loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = (threshold, bet_multiplier)
                best_profit = avg_profit
                best_roi = avg_roi
                best_pct_games_bet = avg_pct_games_bet
                best_total_games_bet = avg_games_bet

    return results, best_params, best_loss, best_profit, best_roi, best_pct_games_bet, best_total_games_bet

# Define parameter grids
threshold_values = np.linspace(2.20, 2.30, 10)  # Adjust range
bet_multipliers = np.linspace(0.8, .9, 10)  # Bet sizing range

# Run Grid Search Cross-Validation
results, best_params, best_loss, best_profit, best_roi, best_pct_games_bet, best_total_games_bet = grid_search_betting_strategy(betting_test, threshold_values, bet_multipliers)

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Threshold", "Bet Multiplier", "Avg Profit", "Avg ROI", "Pct Games Bet", "Total Games Bet", "Loss"])

# Extract best parameters directly from cross-validation
best_threshold, best_bet_multiplier = best_params

# Print Best Parameters
print("\n **Best Parameters Found (Based on Cross-Validation):**")
print(f"   ➡ Best Threshold: {best_threshold:.2f}")
print(f"   ➡ Best Bet Multiplier: {best_bet_multiplier:.2f}")
print(f"   ➡ Best Cross-Validation Loss (Lower is Better): {best_loss:.4f}")

print("\n **Cross-Validation Performance (Averages Over Folds):**")
print(f"   ➡ Avg Profit: {best_profit:.2f}")
print(f"   ➡ Avg ROI: {best_roi:.2f}%")
print(f"   ➡ Avg % of Games Bet: {best_pct_games_bet:.2f}%")
print(f"   ➡ Avg Total Games Bet Per Fold: {best_total_games_bet:.0f}")


# %%
# Step 1: Apply the best betting strategy to the full dataset
betting_results = evaluate_betting_strategy(betting_test, best_threshold, best_bet_multiplier)
betting_df = betting_results["betting_df"]

# Step 2: Compute metrics per week
weekly_stats = betting_df.groupby("week").agg(
    total_games=("week", "count"),
    games_bet=("bet_team", lambda x: (x != "No Bet").sum()),
    total_profit=("profit", "sum"),
    total_wagered=("bet_size", "sum")
).reset_index()

# Ensure Weeks 1-18 are included, even if missing in the data
full_weeks = pd.DataFrame({"week": np.arange(1, 19)})  # Explicitly define weeks 1-18
weekly_stats = full_weeks.merge(weekly_stats, on="week", how="left").fillna(0)  # Fill missing weeks with 0


# Convert to percentage metrics
weekly_stats["pct_games_bet"] = (weekly_stats["games_bet"] / weekly_stats["total_games"]) * 100
weekly_stats["weekly_return"] = (weekly_stats["total_profit"] / weekly_stats["total_wagered"]) * 100

# Ensure 1D conversion
weeks = weekly_stats["week"].values
pct_games_bet = weekly_stats["pct_games_bet"].values
weekly_return = weekly_stats["weekly_return"].values

# Step 3: Plot Fraction of Games Bet Per Week
plt.figure(figsize=(12, 5))
plt.plot(weeks, pct_games_bet, marker='o', linestyle='-', label="Fraction of Games Bet")
plt.xlabel("Week of Season")
plt.ylabel("Percentage of Games Bet (%)")
plt.title("Fraction of Games Bet by Week (Weeks 1-18)")
plt.xticks(np.arange(1, 19, 1))  # Ensure every week is shown on x-axis
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Plot Weekly Return (Profit %)
plt.figure(figsize=(12, 5))
plt.plot(weeks, weekly_return, marker='o', linestyle='-', label="Weekly Return (%)", color="red")
plt.xlabel("Week of Season")
plt.ylabel("Return on Investment (%)")
plt.title("Weekly Return by Week (Weeks 1-18)")
plt.axhline(y=0, color='black', linestyle='--')  # Baseline at 0%
plt.xticks(np.arange(1, 19, 1))  # Ensure every week is shown on x-axis
plt.legend()
plt.grid(True)
plt.show()





# %%
# Compute yearly metrics
yearly_stats = betting_df.groupby("year").agg(
    total_games=("year", "count"),
    games_bet=("bet_team", lambda x: (x != "No Bet").sum()),
    total_profit=("profit", "sum"),
    total_wagered=("bet_size", "sum")
).reset_index()

# Convert to percentage metrics
yearly_stats["pct_games_bet"] = (yearly_stats["games_bet"] / yearly_stats["total_games"]) * 100
yearly_stats["roi"] = (yearly_stats["total_profit"] / yearly_stats["total_wagered"]) * 100

yearly_stats

# %%
# Compute profit, ROI, and number of games bet for and against each team
filtered_betting_df = betting_df[betting_df["bet_team"] != "No Bet"].copy()
# Betting ON each team
team_bet_on_stats = filtered_betting_df.groupby("bet_team").agg(
    games_bet=("bet_team", "count"),
    total_profit=("profit", "sum"),
    total_wagered=("bet_size", "sum")
).reset_index()

# Calculate ROI
team_bet_on_stats["roi"] = (team_bet_on_stats["total_profit"] / team_bet_on_stats["total_wagered"]) * 100

# Convert to dictionary format
team_bet_on_dict = team_bet_on_stats.set_index("bet_team").to_dict(orient="index")

# Create "against_team" column by determining the opponent in each game
filtered_betting_df["against_team"] = filtered_betting_df.apply(
    lambda row: row["away_team"] if row["bet_team"] == row["home_team"] else row["home_team"], axis=1
)

# Betting AGAINST each team
team_bet_against_stats = filtered_betting_df.groupby("against_team").agg(
    games_bet=("against_team", "count"),
    total_profit=("profit", "sum"),
    total_wagered=("bet_size", "sum")
).reset_index()

# Calculate ROI
team_bet_against_stats["roi"] = (team_bet_against_stats["total_profit"] / team_bet_against_stats["total_wagered"]) * 100

# Convert to dictionary format
team_bet_against_dict = team_bet_against_stats.set_index("against_team").to_dict(orient="index")

# Sorting the "bet on" dictionary by total profit
sorted_team_bet_on_dict = dict(sorted(team_bet_on_dict.items(), key=lambda item: item[1]["total_profit"], reverse=True))

# Sorting the "bet against" dictionary by total profit
sorted_team_bet_against_dict = dict(sorted(team_bet_against_dict.items(), key=lambda item: item[1]["total_profit"], reverse=True))

# Convert to DataFrames for display
bet_on_df = pd.DataFrame.from_dict(sorted_team_bet_on_dict, orient='index').reset_index().rename(columns={"index": "team"})
bet_against_df = pd.DataFrame.from_dict(sorted_team_bet_against_dict, orient='index').reset_index().rename(columns={"index": "team"})



bet_on_df

# %%
bet_against_df

# %%
betting_df[betting_df['profit'] == betting_df['profit'].max()]

# %%
betting_df[betting_df['year'] == 2024].loc[betting_df[betting_df['year'] == 2024]['profit'].idxmax()]


# %%


# Assuming `betting_df` is already loaded, filtering the dataframe
filtered_df = betting_df[(betting_df['year'] == 2024) & 
                         ((betting_df['home_team'] == 'DAL') | (betting_df['away_team'] == 'DAL'))]

# Display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

filtered_df


# %%
