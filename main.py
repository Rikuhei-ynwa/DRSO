import os, pdb, warnings, pickle, argparse, time, random
from tqdm import tqdm

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

from socceraction.data.statsbomb import StatsBombLoader
from mplsoccer import Sblocal
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import obso_player as obs
import SPADL_config as spc

warnings.simplefilter('ignore')
pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int)
parser.add_argument("--data", type=str, default="statsbomb")
parser.add_argument("--game", type=str, default="all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--set_vel", type=float, default=0.0)
parser.add_argument("--skip_load_rawdata", action="store_true")
parser.add_argument("--skip_compare_the_number_of_players", action="store_true")
parser.add_argument("--skip_calculate_obso", action="store_true")
parser.add_argument("--skip_verify_obso", action="store_true")
parser.add_argument("--skip_identify_optimal_positioning", action="store_true")
parser.add_argument("--skip_evaluate_team_defense", action="store_true")
parser.add_argument("--skip_show_results", action="store_true")
parser.add_argument("--pickle", type=int, default=5)
args = parser.parse_known_args()[0]

pickle.HIGHEST_PROTOCOL = args.pickle

start = time.time()

random.seed(args.seed)
np.random.seed(args.seed)

datafolder = f"../DRSO_data/data-{args.data}/" + args.game
os.makedirs(datafolder, exist_ok=True)

count = args.count
print("----------------------------------------")
print(f"{count}")
print("----------------------------------------")

### 1. load and convert statsbomb data ###
if args.data == "statsbomb":
    data_folder = "../open-data-20231002/data"
    DLoader = StatsBombLoader(root=data_folder, getter="local")
else:
    pdb.set_trace()

competitions = DLoader.competitions()
set(competitions.competition_name)

if args.game == "wc2022":
    selected_competitions = competitions[
        (competitions.competition_name == "FIFA World Cup") & (competitions.season_name == "2022")
    ]
elif args.game == "euro2020":
    selected_competitions = competitions[
        (competitions.competition_name == "UEFA Euro") & (competitions.season_name == "2020")
    ]
elif args.game == "euro2022":
    selected_competitions = competitions[
        (competitions.competition_name == "UEFA Women's Euro") & (competitions.season_name == "2022")
    ]
else:  # All data
    selected_competitions = competitions[
        ((competitions.competition_name == "FIFA World Cup") & (competitions.season_name == "2022"))
        | ((competitions.competition_name == "UEFA Euro") & (competitions.season_name == "2020"))
        | ((competitions.competition_name == "UEFA Women's Euro") & (competitions.season_name == "2022"))
    ]

# Get games from all selected competitions
games = pd.concat([DLoader.games(row.competition_id, row.season_id) for row in selected_competitions.itertuples()])
pdb.set_trace()

if args.skip_load_rawdata:
    print("loading rawdata is skipped.")
else:
    matches = {}
    actions = {}
    shot_freeze_frames = {}
    lineups = {}

    sbload = Sblocal()
    for game in tqdm(list(games.itertuples()), desc="Loading game data"):
        df_event, _, df_freeze, _ = sbload.event(os.path.join(data_folder, f"events/{game.game_id}.json"))
        df_frame, df_visible = sbload.frame(os.path.join(data_folder, f"three-sixty/{game.game_id}.json"))
        df_lineup = sbload.lineup(os.path.join(data_folder, f"lineups/{game.game_id}.json"))
        actions[game.game_id] = obs.convert_Metrica_for_event(game,df_event,df_frame,df_visible)
        shot_freeze_frames[game.game_id] = df_freeze
        lineups[game.game_id] = df_lineup[
            ["player_id", "player_name", "player_nickname", "jersey_number", "team_id", "team_name"]
            ]

    spadl_h5 = os.path.join(datafolder, f"spadl-{args.data}-{args.game}.h5")
    # Store all spadl data in h5-file
    with pd.HDFStore(spadl_h5, pickle_protocol=5) as spadlstore:
        spadlstore["competitions"] = selected_competitions
        for game_id in actions.keys():
            spadlstore[f"shot_freeze_frames/{game_id}"] = shot_freeze_frames[game_id]
            spadlstore[f"lineups/{game_id}"] = lineups[game_id]
            spadlstore[f"actions/{game_id}"] = actions[game_id]

# Configure file and folder names
spadl_h5 = os.path.join(datafolder, f"spadl-{args.data}-{args.game}.h5")
print("----------------------------------------")


### 1.2. Compare the number of players in attacking-third and in non attacking-third area. ###
if args.skip_compare_the_number_of_players:
    print("Comparing the number of players is skipped.")
else:
    visible_num_pl_in_att_third = []
    visible_num_pl_in_non_att_third = []
    visible_def_gk_in_att_third = []
    visible_def_gk_in_non_att_third = []
    for game in tqdm(list(games.itertuples())):
        actions = pd.read_hdf(spadl_h5, f"actions/{game.game_id}")
        first_kickoff_team = actions.iloc[actions[actions["Period"]==1].index[0]]["Team"]
        second_kickoff_team = actions.iloc[actions[actions["Period"]==2].index[0]]["Team"]
        # actions = actions[actions["Type"]!="Ball Receipt"].reset_index(drop=True)
        for event_num in range(len(actions)):
            action = actions.loc[event_num]
            visible_num_pl = int(np.sum(~np.isnan(action["Freeze Frame 360"])) / 2)
            if action['Type'] in spc.DEFENSE_TYPE:
                visible_def_kp = int(np.sum(~np.isnan(action["Freeze Frame 360"][-2:])) / 2)
            else:
                visible_def_kp = int(np.sum(~np.isnan(action["Freeze Frame 360"][20:22])) / 2)

            att_third = np.all(action[["Start X","End X"]].values >= np.array([17.5,17.5]))
            # shot = ("Shot" in actions["Type"].loc[event_num])

            if att_third:
                visible_num_pl_in_att_third.append(visible_num_pl)
                visible_def_gk_in_att_third.append(visible_def_kp)
            else:
                visible_num_pl_in_non_att_third.append(visible_num_pl)
                visible_def_gk_in_non_att_third.append(visible_def_kp)

    # visualize
    for data1, data2 in zip(
        [visible_num_pl_in_att_third, visible_def_gk_in_att_third],
        [visible_num_pl_in_non_att_third, visible_def_gk_in_non_att_third]
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),sharey=True)
        if data1 == visible_num_pl_in_att_third and data2 == visible_num_pl_in_non_att_third:
            n1, bins1, patches1 = ax1.hist(data1, bins=range(0, 23, 1), align="left")
            ax1.set_title(
                f'(a)The number of visible players in attacking-third\nThe number of event : {len(data1)}',
                fontsize=16,
                )
            n2, bins2, patches2 = ax2.hist(data2, bins=range(0, 23, 1), align="left")
            ax2.set_title(
                f'(b)The number of visible players otherwise\nThe number of event : {len(data2)}',
                fontsize=16,
                )
            max_n = max(max(n1), max(n2))
            ax1.set_ylim(0, max_n+10000)
            ax2.set_ylim(0, max_n+10000)
            m_data1, q1_data1, median_data1, q3_data1, M_data1 = np.percentile(data1, [0,25,50,75,100])
            m_data1, q1_data2, median_data2, q3_data2, M_data2 = np.percentile(data2, [0,25,50,75,100])            
            for i, patch in enumerate(patches1):
                if bins1[i] == q1_data1:
                    patch.set_facecolor('green')
                elif bins1[i] == median_data1:
                    patch.set_facecolor('red')
                elif bins1[i] == q3_data1:
                    patch.set_facecolor('green')
                else:
                    patch.set_facecolor('blue')

            for i, patch in enumerate(patches2):
                if bins2[i] == q1_data2:
                    patch.set_facecolor('green')
                elif bins2[i] == median_data2:
                    patch.set_facecolor('red')
                elif bins2[i] == q3_data2:
                    patch.set_facecolor('green')
                else:
                    patch.set_facecolor('blue')

            for num, bin in zip(n1, bins1):
                if bin in [m_data1, q1_data1, median_data1, q3_data1, 21]:
                    ax1.text(
                        bin+0.1,num+100,int(num),
                        fontsize=12,rotation=70,horizontalalignment="center",
                        )
            for num, bin in zip(n2, bins2):
                if bin in [m_data1, q1_data2, median_data2, q3_data2, 21]:
                    ax2.text(
                        bin+0.2,num+100,int(num),
                        fontsize=12,rotation=70,horizontalalignment="center",
                        )
            ax1.set_xticks(bins1)
            ax2.set_xticks(bins2)
            ax1.tick_params(axis="both",labelsize=12)
            ax2.tick_params(axis="both",labelsize=12)
            # save the figure.
            fig.tight_layout()
            fig.savefig(datafolder + f"/compare_the_number_of_players.png")
            print(datafolder + f"/compare_the_number_of_players.png" + " is saved")

        elif data1 == visible_def_gk_in_att_third and data2 == visible_def_gk_in_non_att_third:
            n1, bins1, _ = ax1.hist(data1, bins=[0,1,2], align="left")
            ax1.set_title(
                f'(a)The number of defense GK in attacking-third\nThe number of event : {len(data1)}',
                fontsize=16,
                )
            n2, bins2, _ = ax2.hist(data2, bins=[0,1,2], align="left")
            ax2.set_title(
                f'(b)The number of defense GK otherwise\nThe number of event : {len(data2)}',
                fontsize=16,
                )
            max_n = max(max(n1), max(n2))
            ax1.set_ylim(0, max_n+100000)
            ax2.set_ylim(0, max_n+100000)

            text_degree1 = [
                ax1.text(
                    bin,num+1000,int(num),
                    fontsize=16,rotation=70,horizontalalignment="center",
                ) for num, bin in zip(n1, bins1) if num
            ]
            text_degree1 = [
                ax2.text(
                    bin,num+1000,int(num),
                    fontsize=16,rotation=70,horizontalalignment="center",
                ) for num, bin in zip(n2, bins2) if num
            ]
            ax1.set_xticks(bins1)
            ax2.set_xticks(bins2)
            ax1.tick_params(axis="both",labelsize=16)
            ax2.tick_params(axis="both",labelsize=16)
            # save the figure.
            fig.tight_layout()
            fig.savefig(datafolder + f"/compare_the_number_of_defense_gk.png")
            print(datafolder + f"/compare_the_number_of_defense_gk.png" + " is saved")

        plt.clf()
        plt.close()
        
    from scipy.spatial import distance
    hist1, _ = np.histogram(visible_num_pl_in_att_third, bins=23, density=True)
    hist2, _ = np.histogram(visible_num_pl_in_non_att_third, bins=23, density=True)
    hist_distance = distance.euclidean(hist1, hist2)
    print(hist_distance)

print("----------------------------------------")


### 2. calculate PPCF and OBSO ###
# load control and transition model
params = mpc.default_model_params()
EPV = mepv.load_EPV_grid('EPV_grid.csv')
EPV = EPV / np.max(EPV)
Trans_df = pd.read_csv('Transition_gauss.csv', header=None)
Trans = np.array((Trans_df))
Trans = Trans / np.max(Trans)
n_grid_cells_x = 50
field_dimen = (105.0, 68.0)

# set OBSO data
if args.skip_calculate_obso:
    print("Calculating obso is skipped.")
else:
    game = games.iloc[(count-1)]
    # for game in tqdm(list(games.itertuples()),desc=f"Calcurating and storing the values of OBSO",):
    print("--------------------")
    print(f"{game.game_id}")
    print("--------------------")
    os.makedirs(datafolder+"/main/obso", exist_ok=True)

    actions = pd.read_hdf(spadl_h5, f"actions/{game.game_id}")
    obso = np.zeros((len(actions), 32, 50))
    ppcf = np.zeros((len(actions), 32, 50))
    first_kickoff_team = actions.iloc[actions[actions["Period"]==1].index[0]]["Team"]
    second_kickoff_team = actions.iloc[actions[actions["Period"]==2].index[0]]["Team"]
    attackers_list = []
    defenders_list = []
    
    for event_num in tqdm(range(len(actions)),desc="Setting OBSO"):
        action = actions.loc[event_num]
        att_third = np.all(action[["Start X","End X"]].values >= np.array([17.5,17.5]))
        # shot = ("Shot" in actions["Type"].loc[event_num])
        if att_third:
            print("event id: ", event_num)

            if action['Period']==1:
                if action['Team']==first_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1
            elif action['Period']==2:
                if action['Team']==second_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1

            elif action['Period']==3:
                if action['Team']==first_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1
            elif action['Period']==4:
                if action['Team']==second_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1

            # get the details of the event (frame, team in possession, ball_start_position)
            actor = action["Actor"]
            actor_team = action["Team"]
            event = action["Type"]
            # correct coordinates so that the kickoff team attacks from left to right.
            ball_start_pos = direction * action[["Start X","Start Y"]].values.astype(float)
            ball_end_pos = direction * action[["End X","End Y"]].values.astype(float)
            duration = action["Duration"]
            coordinates = (direction * action["Freeze Frame 360"]).reshape(-1,2)

            PPCF, _, _, attackers, defenders = mpc.generate_pitch_control_for_event(
                actor,
                actor_team,
                event,
                ball_start_pos,
                ball_end_pos,
                coordinates,
                duration,
                direction, 
                params, 
                optimal=False, 
                offsides=True, 
                set_vel=args.set_vel
                )
            OBSO, _ = obs.calc_obso(
                PPCF, 
                Trans, 
                EPV, 
                ball_start_pos,
                attack_direction=direction
                )
        else:
            attackers = []
            defenders = []
            PPCF = np.zeros((32, 50))
            OBSO = np.zeros((32, 50))

        attackers_list.append(attackers)
        defenders_list.append(defenders)
        ppcf[event_num] = PPCF
        obso[event_num] = OBSO
        
    # save the results.
    obso_data = {"PPCF": ppcf, "OBSO": obso, "attackers": attackers_list, "defenders": defenders_list}
    
    with open(datafolder+"/main/obso"+f"/{game.game_id}_{args.set_vel}.pkl", "wb") as f:
        pickle.dump(obso_data, f)

print("----------------------------------------")


### 3. Verify sample OBSO. ###
if args.skip_verify_obso:
    print("Verifying obso is skipped.")
else:
    team_scorers = {}
    team_non_scorers = {}
    game_scorers = {}
    game_non_scorers = {}
    players_rmses = []
    
    for game in tqdm(list(games.itertuples()),desc=f"Calculating metrics and storing the results",):
        actions = pd.read_hdf(spadl_h5, f"actions/{game.game_id}")
        lineups = pd.read_hdf(spadl_h5, f"lineups/{game.game_id}")
        
        home_team_name = lineups.loc[lineups["team_id"]==game.home_team_id,"team_name"].values[0]
        away_team_name = lineups.loc[lineups["team_id"]==game.away_team_id,"team_name"].values[0]

        with open(datafolder+"/main/obso"+f"/{game.game_id}_{args.set_vel}.pkl", 'rb') as f:
            obso_data = pickle.load(f)
        
        scorers_se = []
        non_scorers_se = []

        obsos, attackers_list = obso_data["OBSO"], obso_data["attackers"]

        for event_num in range(len(actions)):
            action = actions.loc[event_num]
            att_third = np.all(action[["Start X","End X"]].values >= np.array([17.5,17.5]))
            
            if att_third and action["Period"]!=5:
                shot = ("shot" in action["Type"]) and (action["Type"] != "penalty_shot")
                if shot:
                    obso = obsos[event_num]
                    attackers = attackers_list[event_num]
                    players_se, scorer_se, non_scorer_se = obs.verify_obso(
                        action, obso, attackers, n_grid_cells_x,
                    )

                    # players_rmse
                    players_rmses.append(np.sqrt(np.nanmean(np.array(players_se))))
                    # teams_rmse
                    if action["Team"] == "Home":
                        if home_team_name in team_scorers:
                            team_scorers[home_team_name].append(scorer_se)
                            team_non_scorers[home_team_name].append(non_scorer_se)
                        else:
                            team_scorers[home_team_name] = []
                            team_non_scorers[home_team_name] = []
                            team_scorers[home_team_name].append(scorer_se)
                            team_non_scorers[home_team_name].append(non_scorer_se)
                    elif action["Team"] == "Away":
                        if away_team_name in team_scorers:
                            team_scorers[away_team_name].append(scorer_se)
                            team_non_scorers[away_team_name].append(non_scorer_se)
                        else:
                            team_scorers[away_team_name] = []
                            team_non_scorers[away_team_name] = []
                            team_scorers[away_team_name].append(scorer_se)
                            team_non_scorers[away_team_name].append(non_scorer_se)
                    # players_rmse
                    scorers_se.append(scorer_se)
                    non_scorers_se.append(non_scorer_se)
                    
        game_scorers[game.game_id] = np.sqrt(np.nanmean(np.array(scorers_se)))
        game_non_scorers[game.game_id] = np.sqrt(np.nanmean(np.array(non_scorers_se)))
    
    # scorers/non_scorers rmse for each team and for each game
    team_scorers = {team: team_scorers[team] for team in sorted(team_scorers)}
    team_non_scorers = {team: team_non_scorers[team] for team in sorted(team_non_scorers)}
    for team in team_scorers.keys():
        team_scorers[team] = np.sqrt(np.nanmean(np.array(team_scorers[team])))
        team_non_scorers[team] = np.sqrt(np.nanmean(np.array(team_non_scorers[team])))
    
    metrics_game_team = {
        "teams": {"scorers": team_scorers, "non_scorers": team_non_scorers},
        "games": {"scorers": game_scorers, "non_scorers": game_non_scorers},
    }
    os.makedirs(datafolder+"/main/metrics", exist_ok=True)
    with open(datafolder+f"/main/metrics/metrics_game_team_{args.set_vel}.pkl", "wb") as f:
        pickle.dump(metrics_game_team, f)
    print(datafolder+f"/main/metrics/metrics_game_team_{args.set_vel}.pkl" + " is saved.")

    # calculate verification results
    players_mean = np.nanmean(players_rmses)
    players_std = np.nanstd(players_rmses)

    team_scorers_res = np.empty(len(metrics_game_team["teams"]["scorers"]))
    team_non_scorers_res = np.empty(len(metrics_game_team["teams"]["non_scorers"]))
    for i, team in enumerate(metrics_game_team["teams"]["scorers"]):
        team_scorers_res[i] = metrics_game_team["teams"]["scorers"][team]
        team_non_scorers_res[i] = metrics_game_team["teams"]["non_scorers"][team]
    team_scorers_mean = np.nanmean(team_scorers_res)
    team_scorers_std = np.nanstd(team_scorers_res)
    team_non_scorers_mean = np.nanmean(team_non_scorers_res)
    team_non_scorers_std = np.nanstd(team_non_scorers_res)

    game_scorers_res = np.empty(len(metrics_game_team["games"]["scorers"]))
    game_non_scorers_res = np.empty(len(metrics_game_team["games"]["non_scorers"]))
    for i, game in enumerate(metrics_game_team["games"]["scorers"]):
        game_scorers_res[i] = metrics_game_team["games"]["scorers"][game]
        game_non_scorers_res[i] = metrics_game_team["games"]["non_scorers"][game]
    game_scorers_mean = np.nanmean(game_scorers_res)
    game_scorers_std = np.nanstd(game_scorers_res)
    game_non_scorers_mean = np.nanmean(game_non_scorers_res)
    game_non_scorers_std = np.nanstd(game_non_scorers_res)

    metrics_result = {
        "players": {"mean": players_mean, "std": players_std},
        "teams": {
            "scorers": {"mean": team_scorers_mean, "std": team_scorers_std},
            "non_scorers": {"mean": team_non_scorers_mean, "std": team_non_scorers_std}
            },
        "games": {
            "scorers": {"mean": game_scorers_mean, "std": game_scorers_std},
            "non_scorers": {"mean": game_non_scorers_mean, "std": game_non_scorers_std}
            },
    }
    with open(datafolder+f"/main/metrics/metrics_result_{args.set_vel}.pkl", "wb") as f:
        pickle.dump(metrics_result, f)
    print(datafolder+f"/main/metrics/metrics_result_{args.set_vel}.pkl" + " is saved.")

print("----------------------------------------")


### 4. Identify optimal positioning strategies. ###
if args.skip_identify_optimal_positioning:
    print("Identifying optimal positionings is skipped")
else:
    game = games.iloc[(count-1)]
    # for game in tqdm(list(games.itertuples())):
    print("--------------------")
    print(f"{game.game_id}")
    print("--------------------")
    with open(datafolder+"/main/obso"+f"/{game.game_id}_{args.set_vel}.pkl", 'rb') as f:
        obso_data = pickle.load(f)

    actions = pd.read_hdf(spadl_h5, f"actions/{game.game_id}")

    for event_num in tqdm(range(len(actions)),desc="Identifying optimal positioning strategies"):
        action = actions.loc[event_num]
        first_kickoff_team = actions.iloc[actions[actions["Period"]==1].index[0]]["Team"]
        second_kickoff_team = actions.iloc[actions[actions["Period"]==2].index[0]]["Team"]

        att_third = np.all(action[["Start X","End X"]].values >= np.array([17.5,17.5]))
        if att_third:
            optimal_positioning_at_event = {"event_num": event_num,}

            if action['Period']==1:
                if action['Team']==first_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1
            elif action['Period']==2:
                if action['Team']==second_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1

            elif action['Period']==3:
                if action['Team']==first_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1
            elif action['Period']==4:
                if action['Team']==second_kickoff_team:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = -1
                    else:
                        direction = 1
                else:
                    if (action['Type'] in spc.DEFENSE_TYPE) or (action['Type'] in spc.KEEPER_SPECIFIC_TYPE):
                        direction = 1
                    else:
                        direction = -1

            obso = obso_data["OBSO"][event_num]
            attackers = obso_data["attackers"][event_num]
            defenders = obso_data["defenders"][event_num]

            # get the details of the event (frame, team in possession, ball_start_position)
            actor = action["Actor"]
            actor_team = action["Team"]
            event = action["Type"]
            # correct coordinates so that the kickoff team attacks from left to right.
            ball_start_pos = direction * action[["Start X","Start Y"]].values.astype(float)
            ball_end_pos = direction * action[["End X","End Y"]].values.astype(float)
            duration = action["Duration"]
            coordinates = (direction * action["Freeze Frame 360"]).reshape(-1,2)

            optimal_positioning_at_event["result"] = obs.identify_optimal_positionings(
                actor,
                actor_team,
                event,
                ball_start_pos,
                ball_end_pos,
                coordinates,
                duration,
                obso, 
                attackers, 
                defenders, 
                direction, 
                params, 
                Trans, 
                EPV, 
                set_vel = args.set_vel,
            )

            # save the result
            os.makedirs(datafolder+f"/main/optimal_positioning/{game.game_id}", exist_ok=True)
            with open(datafolder+f"/main/optimal_positioning/{game.game_id}"+f"/{event_num}.pkl", "wb") as f:
                pickle.dump(optimal_positioning_at_event, f)
            print(datafolder+f"/main/optimal_positioning/{game.game_id}"+f"/{event_num}.pkl" + " is saved.")


### 5. Evaluate team defense ###
# if args.skip_evaluate_team_defense:
#     print("Evaluating team defense is skipped")
# else:
#     # all seasons
#     team_drso_results_all = {}
#     for game in tqdm(list(games.itertuples()),desc="Evaluating team defense during all seasons"):
#         sbload = Sblocal()
#         actions = pd.read_hdf(spadl_h5, f"actions/{game.game_id}")
#         lineups = pd.read_hdf(spadl_h5, f"lineups/{game.game_id}")

#         home_team_name = lineups.loc[lineups["team_id"]==game.home_team_id,"team_name"].values[0]
#         away_team_name = lineups.loc[lineups["team_id"]==game.away_team_id,"team_name"].values[0]
#         home_team_differences = []
#         home_team_concedes = game.away_score
#         away_team_differences = []
#         away_team_concedes = game.home_score

#         for event_num in range(len(actions)):
#             action = actions.loc[event_num]
#             att_third = np.all(action[["Start X","End X"]].values >= np.array([17.5,17.5]))

#             if att_third and action['Period']!=5:
#                 with open(datafolder+f"/main/optimal_positioning/{game.game_id}/{event_num}.pkl", "rb") as f:
#                     optimal_positioning = pickle.load(f)

#                 if len(optimal_positioning["result"]) == 0:
#                     continue
#                 else:
#                     Differences = []
#                     for i in range(len(optimal_positioning["result"])):
#                         Difference = (
#                             optimal_positioning["result"][i]["optimal"]["obso"] - optimal_positioning["result"][i]["data"]["obso"]
#                             )
#                         Differences.append(Difference)
#                     if (action["Type"] in spc.DEFENSE_TYPE) or (action["Type"] in spc.KEEPER_SPECIFIC_TYPE):
#                         if action["Team"] == "Home":
#                             home_team_differences.append(np.nanmean(np.array(Differences)))
#                         elif action["Team"] == "Away":
#                             away_team_differences.append(np.nanmean(np.array(Differences)))
#                     else:
#                         if action["Team"] == "Home":
#                             away_team_differences.append(np.nanmean(np.array(Differences)))
#                         elif action["Team"] == "Away":
#                             home_team_differences.append(np.nanmean(np.array(Differences)))

#         home_team_result = {"Diff": home_team_differences}
#         away_team_result = {"Diff": away_team_differences}

#         if home_team_name in team_drso_results_all:
#             team_drso_results_all[home_team_name]["Diff"].extend(home_team_result["Diff"])
#         else:
#             team_drso_results_all[home_team_name] = {}
#             team_drso_results_all[home_team_name]["Diff"] = home_team_result["Diff"]
#         if away_team_name in team_drso_results_all:
#             team_drso_results_all[away_team_name]["Diff"].extend(away_team_result["Diff"])
#         else:
#             team_drso_results_all[away_team_name] = {}
#             team_drso_results_all[away_team_name]["Diff"] = away_team_result["Diff"]

#     team_drso_results = {
#         "all": team_drso_results_all,
#     }

#     with open(datafolder+f"/main/optimal_positioning/team_drso_results.pkl", "wb") as f:
#         pickle.dump(team_drso_results, f)
#     print(datafolder+f"/main/optimal_positioning/team_drso_results.pkl" + " is saved.")


# if args.skip_show_results:
#     print("Showing results is skipped.")
# else:
#     with open(datafolder+f"/main/optimal_positioning/team_drso_results.pkl", "rb") as f:
#         team_drso_results = pickle.load(f)

#     key = "all"
#     team_differences_list =[team_drso_results[key][team]["Diff"] for team in team_drso_results[key].keys()]
#     team_concedes_list = [team_drso_results[key][team]["concedes"] for team in team_drso_results[key].keys()]
#     correlation_cd, p_value_cd = scipy.stats.pearsonr(team_concedes_list, team_differences_list)
#     print(correlation_cd,p_value_cd)

#     fig, ax = plt.subplots(1, 1, figsize=(12, 9))
#     ax.scatter(team_concedes_list,team_differences_list,marker="o",s=50,color="blue",alpha=0.6)
#     ax.set_xlim(0, max(team_concedes_list)+1)
#     text = [ax.text(
#                 team_drso_results[key][team]["concedes"],
#                 team_drso_results[key][team]["Diff"],
#                 team,
#                 fontsize=24,
#                 color="#595959",
#             ) for team in team_drso_results[key].keys()]
#     adjust_text(text, arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6,))
#     ax.tick_params(axis="both",colors="#595959",labelsize=20,grid_color="#595959",grid_alpha=0.3,)
#     ax.grid()
#     fig.savefig(os.path.join(datafolder+f"/main/team_drso_results_for_concedes.png"))
#     print(os.path.join(datafolder+f"/main/team_drso_results_for_concedes.png") + " is saved")
#     plt.clf()
#     plt.close()


pdb.set_trace()