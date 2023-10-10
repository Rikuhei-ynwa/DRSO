#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:19 2020

Module for calculating a Pitch Control surface using MetricaSports's tracking & event data.

Pitch control (at a given location on the field) is the probability that a team will gain
possession if the ball is moved to that location on the field.

Methdology is described in "Off the ball scoring opportunities" by William Spearman:
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Functions
----------

calculate_pitch_control_at_target(): calculate the pitch control probability for the attacking and defending teams at a specified target position on the ball.

generate_pitch_control_for_event(): this function evaluates pitch control surface over the entire field at the moment
of the given event (determined by the index of the event passed as an input)

Classes
---------

The 'player' class collects and stores trajectory information for each player required by the pitch control calculations.

@author: Laurie Shaw (@EightyFivePoint)

"""

import numpy as np
from scipy.spatial.distance import cdist
# import pandas as pd
import SPADL_config as spc


def initialise_players(
    actor_team: str,
    event: str,
    actor: int,
    coordinates: np.ndarray, 
    ball_info: dict, 
    direction: int, 
    params: dict,
    set_vel: float,
    ) -> list:
    """
    initialise_players(positions,teamname,params)

    create a list of player objects that holds their positions and velocities from the tracking data dataframe

    Parameters
    -----------

    event: Metrica event dataframe 
    direction: The attacking directions of team that a player belong to
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returns
    -----------

    attackers: list of attacker objects for the team at at given instant
    defenders: list of defender objects for the team at at given instant

    """

    # create list
    teammate_players = []
    opponent_players = []

    # Setting
    goal = direction * np.array([52.5,0])
    # goal = direction * np.array([[52.5,-3.66],[52.5,3.66]])

    ### Extract players with special speed ###
    if event in spc.DEFENSE_TYPE or event in spc.KEEPER_SPECIFIC_TYPE:
        d0_index = actor
        d0 = coordinates[d0_index]
        a0_index = cdist([d0], coordinates[:11]).argsort()[0][1]
        a0 = coordinates[:11][a0_index]
    else:
        a0_index = actor
        a0 = coordinates[a0_index]
        d0_index = cdist([a0], coordinates[:11]).argsort()[0][1]
        d0 = coordinates[:11][d0_index]

    indices ={"a0" : a0_index, "d0" : d0_index}

    for pl in range(22):
        team_player = player(pl, actor_team, event, actor, ball_info, coordinates, goal, indices, params, set_vel)
        if pl < 11:
            opponent_players.append(team_player)
        elif 11 <= pl < 22:
            teammate_players.append(team_player)

    if event in spc.DEFENSE_TYPE or event in spc.KEEPER_SPECIFIC_TYPE:
        attackers = opponent_players
        defenders = teammate_players
    else:
        attackers = teammate_players
        defenders = opponent_players

    return attackers, defenders


def initialise_players_for_optimal(
    event: str,
    actor: str,
    ball_info: dict,
    direction: int,
    attackers: list,
    defenders: list,
    set_vel: float,
    ) -> list:
    """
    initialise_players_v2(event: pd.Series, ball_info: dict, direction: int, params: dict)

    create a list of player objects that holds their positions and velocities from the tracking data dataframe

    Parameters
    -----------

    event: Metrica event dataframe 
    direction: The attacking directions of team that a player belong to
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returns
    -----------

    attackers: list of attackers objects for the team at at given instant
    defenders: list of defenders objects for the team at at given instant

    """

    # Setting
    ball_end_pos = ball_info["ball_end_pos"]
    goal = direction * np.array([52.5,0])

    ### Extract players with special speed ###
    if event in spc.DEFENSE_TYPE or event in spc.KEEPER_SPECIFIC_TYPE:
        for defender in defenders:
            if defender.id == actor:
                d0_index = defender.id
                break
    else:
        for attacker in attackers:
            if attacker.id == actor:
                a0 = attacker.position
                break
        for i, defender in enumerate(defenders):
            if i == 0:
                min_dist = np.sqrt((defender.position[0]-a0[0])**2 + (defender.position[1]-a0[1])**2)
                d0_index = defender.id
            else:
                dist = np.sqrt((defender.position[0]-a0[0])**2 + (defender.position[1]-a0[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    d0_index = defender.id

    def _get_velosity(
            ball_end_pos: np.array, coordinate: np.array, goal: np.array, spc_index: int, pl_index: int, set_vel: float,
            ):
        if set_vel == 0.0:
            if np.any(np.isnan(coordinate)):
                return np.array([np.nan, np.nan])
            else:
                return np.array([0.0, 0.0])
        else:
            if np.any(np.isnan(coordinate)):
                return np.array([np.nan, np.nan])
            elif pl_index == spc_index:
                return (
                    (ball_end_pos - coordinate) / np.sqrt((ball_end_pos[0] - coordinate[0]) ** 2 + (ball_end_pos[1] - coordinate[1]) ** 2)
                    ) * set_vel # variable
            else:
                return (
                    (goal - coordinate) / np.sqrt((goal[0] - coordinate[0]) ** 2 + (goal[1] - coordinate[1]) ** 2)
                    ) * set_vel # variable

    for i, defender_pl in enumerate(defenders):
        defender_pl.velocity = _get_velosity(ball_end_pos, defender_pl.position, goal, d0_index, defender_pl.id, set_vel)
        defenders[i] = defender_pl

    return attackers, defenders


def check_offsides(
    actor: int,
    event: str,
    attackers: list,
    defenders: list,
    ball_position: np.ndarray,
    direction: int,
    tol=0.2,
    ) -> list:
    """
    check_offsides( attackers, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):

    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from
    the 'attackers' list and ignored in the pitch control calculation.

    Parameters
    -----------
        attackers: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defenders: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m

    Returns
    -----------
        attackers: list of 'player' objects for the players on the attacking team with offside players removed
    """

    # make sure defending goalkeeper is actually on the field!
    if np.any(np.isnan(defenders[10].position)):
        return []
    else:
        # 1. find the x-position of the second-deepest defeending player (including GK)
        # 2. define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
        # (3. any attacking players with x-position greater than the offside line are offside)
        attackers_tmp = []
        if direction == 1:
            second_deepest_defender_x = sorted(
                [defender.position[0] for defender in defenders],
                reverse=True,
            )[1]
            offside_line = max(second_deepest_defender_x, ball_position[0], 0.0) + tol
            for attacker in attackers:
                if event in spc.DEFENSE_TYPE or event in spc.KEEPER_SPECIFIC_TYPE:
                    if attacker.position[0] > offside_line:
                        continue
                    else:
                        attackers_tmp.append(attacker)
                else:
                    if attacker.id == actor:
                        attackers_tmp.append(attacker)
                    else:
                        if attacker.position[0] > offside_line:
                            continue
                        else:
                            attackers_tmp.append(attacker)

        elif direction == -1:
            second_deepest_defender_x = sorted(
                [defender.position[0] for defender in defenders],
            )[1]
            offside_line = min(second_deepest_defender_x, ball_position[0], 0.0) - tol
            for attacker in attackers:
                if event in spc.DEFENSE_TYPE or event in spc.KEEPER_SPECIFIC_TYPE:
                    if attacker.position[0] < offside_line:
                        continue
                    else:
                        attackers_tmp.append(attacker)
                else:
                    if attacker.id == actor:
                        attackers_tmp.append(attacker)
                    else:
                        if attacker.position[0] < offside_line:
                            continue
                        else:
                            attackers_tmp.append(attacker)
        
        return attackers_tmp


class player(object):
    """
    player() class

    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player

    __init__ Parameters
    -----------
    pl: id (jersey number) of player by str
    team: row (i.e. instant) of either the home or away team tracking Series
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )


    methods include:
    -----------
    simple_time_to_intercept(r_final_pos): time take for player to get to target position (r_final_pos) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept

    """

    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(
            self, pl: int, actor_team: str, event: str, actor: int, ball_info: dict, coordinates: np.array, goal: np.array, indices: dict, params: dict, set_vel: float,
        ):
        self.id = pl
        self.get_teamname(actor_team)
        self.vmax = params["max_player_speed"]  # player max speed in m/s. Could be individualised
        self.amax = params["max_player_accel"]  # player max acceleration in m/s^2. Could be individualised
        self.reaction_time = params["reaction_time"]  # player reaction time in 's'. Could be individualised
        self.tti_sigma = params["tti_sigma"]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params["lambda_att"]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params["lambda_gk"] if self.id == 10 or self.id == 21 else params["lambda_def"]  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(coordinates)
        self.get_velocity(event, ball_info, actor, goal, indices, set_vel,)
        self.PPCF = 0.0  # initialise this for later

    def get_teamname(self, actor_team: str):
        if self.id < 11:
            self.teamname = np.where(np.array(["Home", "Away"]) != actor_team)[0]
        elif 11 <= self.id < 22:
            self.teamname = actor_team

    def get_position(self, coordinates: np.array):
        self.position = coordinates[self.id]
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, event: str, ball_info: dict, actor: int, goal: np.array, indices: dict, set_vel: float,):
        # Regarding players' velocities, we give some candidates
        # self.velocity = np.array([team[self.playername + "vx"], team[self.playername + "vy"]])
        if set_vel == 0.0:
            if np.any(np.isnan(self.position)):
                self.velocity = np.array([np.nan, np.nan])
            else:
                self.velocity = np.array([0.0, 0.0])
        elif set_vel == 0.01:
            if np.any(np.isnan(self.position)):
                self.velocity = np.array([np.nan, np.nan])
            elif (indices["a0"] == actor) and self.id == actor:
                self.velocity = ball_info["ball_velocity"]
                for x in [
                    "pass","shot","offensive_ball_recovery","offensive_foul","dispossessed","miscontrol",
                    ]:
                    if x in event:
                        self.velocity = np.array([0.0,0.0])
            # defensive actor including keeper
            elif (indices["d0"] == actor) and self.id == actor:
                self.velocity = ball_info["ball_velocity"]
            # diffensive_keeper and not actor
            elif (not event in spc.DEFENSE_TYPE) and (not event in spc.KEEPER_SPECIFIC_TYPE) and self.id == 10:
                if "shot" in event:
                    if "Penalty" in event:
                        self.velocity = np.array([0.0,0.0])
                    elif not np.any(np.isnan(self.position)) and (ball_info["ball_end_pos"][1] >= self.position[1]):
                        self.velocity = np.array([0.0,2.0])
                    elif not np.any(np.isnan(self.position)) and (ball_info["ball_end_pos"][1] < self.position[1]):
                        self.velocity = np.array([0.0,-2.0])
                elif self.inframe:
                    self.velocity = np.array([0.0,0.0])
            elif (event in spc.DEFENSE_TYPE) and (event in spc.KEEPER_SPECIFIC_TYPE) and self.id == 21:
                self.velocity = np.array([0.0,0.0])
            else:
                self.velocity = np.array([0.0,0.0])
        else:
            if np.any(np.isnan(self.position)):
                self.velocity = np.array([np.nan, np.nan])
            elif (indices["a0"] == actor) and self.id == actor:
                self.velocity = ball_info["ball_velocity"]
                for x in [
                    "pass","shot","offensive_ball_recovery","offensive_foul","dispossessed","miscontrol",
                    ]:
                    if x in event:
                        self.velocity = np.array([0.0,0.0])
            # defensive actor including keeper
            elif (indices["d0"] == actor) and self.id == actor:
                self.velocity = ball_info["ball_velocity"]
            # diffensive_keeper and not actor
            elif (not event in spc.DEFENSE_TYPE) and (not event in spc.KEEPER_SPECIFIC_TYPE) and self.id == 10:
                if "shot" in event:
                    if "Penalty" in event:
                        self.velocity = np.array([0.0,0.0])
                    elif not np.any(np.isnan(self.position)) and (ball_info["ball_end_pos"][1] >= self.position[1]):
                        self.velocity = np.array([0.0,2.0])
                    elif not np.any(np.isnan(self.position)) and (ball_info["ball_end_pos"][1] < self.position[1]):
                        self.velocity = np.array([0.0,-2.0])
                elif self.inframe:
                    self.velocity = np.array([0.0,0.0])
            elif (event in spc.DEFENSE_TYPE) and (event in spc.KEEPER_SPECIFIC_TYPE) and self.id == 21:
                self.velocity = np.array([0.0,0.0])
            # defender 0 
            elif self.id == indices["d0"]:
                self.velocity = (
                    (ball_info["ball_end_pos"] - self.position)
                    / np.sqrt((ball_info["ball_end_pos"][0] - self.position[0]) ** 2 + (ball_info["ball_end_pos"][1] - self.position[1]) ** 2)
                    * set_vel # variable
                    )
            else:
                self.velocity = (
                    (goal - self.position)
                    / np.sqrt((goal[0] - self.position[0]) ** 2 + (goal[1] - self.position[1]) ** 2)
                    * set_vel # variable
                    )

    def simple_time_to_intercept(self, r_final_pos: np.array):
        self.PPCF = 0.0  # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.

        r_reaction_pos = self.position + self.velocity * self.reaction_time
        movement_time = np.linalg.norm(r_final_pos - r_reaction_pos) / self.vmax
        self.time_to_intercept = self.reaction_time + movement_time

        return self.time_to_intercept

    def probability_intercept_ball(self, T: float):
        # probability of a player arriving at target location at time 'T' given
        # their expected time_to_intercept (time of arrival), as described in Spearman 2018.
        # this function is similar to sigmoid function.

        increate_dataset_to_function = np.pi * ((T - self.time_to_intercept) / (np.sqrt(3.0) * self.tti_sigma))
        f = 1 / (1.0 + np.exp(-increate_dataset_to_function))

        return f


""" Generate pitch control map """


def default_model_params(time_to_control_veto=3) -> dict:
    """
    default_model_params()

    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.

    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control
    it is less than 10^-time_to_control_veto, ignore that player.


    Returns
    -----------

    params: dictionary of parameters required to determine and calculate the model

    """
    # key parameters for the model, as described in Spearman 2018
    params = {}

    # model parameters
    # maximum player acceleration m/s/s, not used in this implementation
    params["max_player_accel"] = 7.0

    # maximum player speed m/s
    params["max_player_speed"] = 5.0

    # seconds taken for player to react and change trajectory. Roughly determined as vmax/amax
    params["reaction_time"] = 0.7

    # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params["tti_sigma"] = 0.45

    # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball,
    # I have set to 1 so that home & away players have same ball control probability
    params["kappa_def"] = 1.0

    # ball control parameter for attacking team
    params["lambda_att"] = 4.3

    # ball control parameter for defending team
    params["lambda_def"] = 4.3 * params["kappa_def"]

    # make goal keepers must quicker to control ball (because they can catch it)
    params["lambda_gk"] = params["lambda_def"] * 3.0

    # average ball travel speed in m/s
    params["average_ball_speed"] = 15.0

    # numerical parameters for model evaluation
    # integration timestep (dt)
    params["int_dt"] = 0.04

    # upper limit on integral time
    params["max_int_time"] = 10

    # assume convergence when PPCF>0.99 at a given location.
    params["model_converge_tol"] = 0.01

    # The following are 'short-cut' parameters. We do not need to calculate PPCF explicitly
    # when a player has a sufficient head start. A sufficient head start is when a player
    # arrives at the target location at least 'time_to_control' seconds before the next player.
    params["time_to_control_att"] = (
        time_to_control_veto * np.log(10) * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto * np.log(10) * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )
    return params


def generate_pitch_control_for_event(
    actor: str,
    actor_team: str,
    event: str,
    ball_start_pos: np.ndarray,
    ball_end_pos: np.ndarray,
    coordinates: np.ndarray,
    duration,
    direction: int,
    params: dict,
    attackers=None,
    defenders=None,
    field_dimen=(
        105.0,
        68.0,
    ),
    n_grid_cells_x=50,
    optimal=False,
    offsides=True,
    set_vel=0.0,
):
    """generate_pitch_control_for_event

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data.
        direction: actor team's attacking direction.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.

    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')

    Returns
    -----------
        PPCF_attack: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCF_attack.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    if duration != 0:
        ball_speed = np.linalg.norm(ball_end_pos - ball_start_pos) / duration
        ball_velocity = (ball_end_pos - ball_start_pos) / duration
    else:
        ball_speed = 0
        ball_velocity = np.array([0.0,0.0])
    ball_info ={
        "ball_start_pos": ball_start_pos,
        "ball_end_pos": ball_end_pos,
        "ball_speed": ball_speed,
        "ball_velocity": ball_velocity,
    }
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0
    # initialise pitch control grids for attackers and defenders
    PPCF_attack = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCF_defense = np.zeros(shape=(len(ygrid), len(xgrid)))
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if actor_team == "Home" or actor_team == "Away":
        if optimal:
            attackers, defenders = initialise_players_for_optimal(
                event,
                actor,
                ball_info,
                direction,
                attackers,
                defenders,
                set_vel,
                )
        else:
            attackers, defenders = initialise_players(
                actor_team,
                event,
                actor,
                coordinates,
                ball_info,
                direction,
                params,
                set_vel
                )
    else:
        assert False, "Team in possession must be either home or away"

    # find any attackers that are offside and remove them from the pitch control calculation
    if offsides:
        attackers = check_offsides(
            actor, event, attackers, defenders, ball_start_pos, direction
            )
    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCF_attack[i, j], PPCF_defense[i, j] = calculate_pitch_control_at_target(
                target_position,
                attackers,
                defenders,
                ball_info,
                event,
                params,
            )
    # # check probabilitiy sums within convergence
    # checksum = np.sum(PPCF_attack + PPCF_defense) / float(n_grid_cells_y * n_grid_cells_x)
    # assert 1 - checksum < params["model_converge_tol"], "Checksum failed: %1.3f" % (1 - checksum)
    return PPCF_attack, xgrid, ygrid, attackers, defenders


def calculate_pitch_control_at_target(
    target_position: np.ndarray,
    attackers: list,
    defenders: list,
    ball_info: dict,
    event: str,
    params: dict,
):
    """calculate_pitch_control_at_target

    Calculates the pitch control probability for the actor and opponent teams at a specified target position on the ball.

    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attackers: list of 'player' objects (see player class above) for the players on the actor's team (team in possession)
        defenders: list of 'player' objects (see player class above) for the players on the opponent team
        ball_info: the ball information including start position, end position, ball speed, and ball velocity. If set to NaN, function will assume that the ball is already at the target position.
        event: event that happened at that time.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returns
    -----------
        PPCFatt: Pitch control probability for the actor team
        PPCFdef: Pitch control probability for the opponent team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

    """

    # calculate ball travel time from start position to end position.
    if (ball_info["ball_start_pos"] is None 
        or np.any(np.isnan(ball_info["ball_start_pos"])) 
        or ball_info["ball_speed"] == 0
        ):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_dist = np.linalg.norm(target_position - ball_info["ball_start_pos"])
        ball_travel_time = ball_travel_dist / ball_info["ball_speed"]
    travel_direction = (target_position - ball_info["ball_start_pos"]) / np.linalg.norm(target_position - ball_info["ball_start_pos"])
    
    # change the way PPCF is obtained depending on the event
    if event in ["ground_pass", "low_pass", "carry"]: # Low Crosses are verified after the conference.
        # set up integration arrays
        dT_array = np.arange(
            params['int_dt'], 
            ball_travel_time + params['int_dt'],
            params["int_dt"],
        )
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1

        while np.sum(PPCFatt) + np.sum(PPCFdef) < 1 - params["model_converge_tol"] and i < dT_array.size:
            # loaction of Control (Spearman 2017)
            location = ball_info["ball_start_pos"] + ball_info["ball_speed"] * dT_array[i-1] * travel_direction

            # solve pitch control model by integrating equation 3 in Spearman et al.
            # first get arrival time of 'nearest' actors' player (nearest also dependent on current velocity)
            if len(attackers) == 0:
                break
            else:
                tau_min_att = np.nanmin([p.simple_time_to_intercept(location) for p in attackers])
                tau_min_def = np.nanmin([p.simple_time_to_intercept(location) for p in defenders])
                tau_min = np.nanmin([tau_min_att, tau_min_def])
            # first remove any player that is far (in time) from the target location
            attackers2 = [p for p in attackers[1:] if p.time_to_intercept - tau_min < params['time_to_control_att']]
            defenders2 = [p for p in defenders if p.time_to_intercept - tau_min < params['time_to_control_def']]

            for player in attackers2:
                dPPCFdT = (1 - PPCFatt[i-1] - PPCFdef[i-1]) * player.probability_intercept_ball(dT_array[i-1]) * player.lambda_att
                assert dPPCFdT >= 0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFatt[i] += dPPCFdT * params['int_dt']

            for player in defenders2:
                dPPCFdT = (1 - PPCFatt[i-1] - PPCFdef[i-1]) * player.probability_intercept_ball(dT_array[i-1]) * player.lambda_def
                assert dPPCFdT >= 0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT * params['int_dt']
                PPCFdef[i] += dPPCFdT * params['int_dt']
            i += 1
        if i >= dT_array.size:
            print(f"Integration failed to converge: {np.sum(PPCFatt)+np.sum(PPCFdef)}")
        
        return np.sum(PPCFatt), np.sum(PPCFdef)
    
    else:
        # set up integration arrays
        dT_array = np.arange(
            ball_travel_time - params["int_dt"],
            ball_travel_time + params["max_int_time"],
            params["int_dt"],
        )
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # first get arrival time of 'nearest' actors' player (nearest also dependent on current velocity)
        if len(attackers) == 0:
            return 0.0, 0.0
        else:
            tau_min_att = np.nanmin([player.simple_time_to_intercept(target_position) for player in attackers])
            tau_min_def = np.nanmin([player.simple_time_to_intercept(target_position) for player in defenders])

            # check whether we actually need to solve equation 3
            if tau_min_att - max(ball_travel_time, tau_min_def) >= params["time_to_control_def"]:
                # if opponent team can arrive significantly before actor team,
                # no need to solve pitch control model
                return 0.0, 1.0
            elif tau_min_def - max(ball_travel_time, tau_min_att) >= params["time_to_control_att"]:
                # if actor team can arrive significantly before opponent team,
                # no need to solve pitch control model
                return 1.0, 0.0
            else:
                # solve pitch control model by integrating equation 3 in Spearman et al.
                # first remove any player that is far (in time) from the target location
                attackers = [p for p in attackers if p.time_to_intercept - tau_min_att < params["time_to_control_att"]]
                defenders = [p for p in defenders if p.time_to_intercept - tau_min_def < params["time_to_control_def"]]
                # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
                ptot = 0.0
                i = 1
                while 1 - ptot > params["model_converge_tol"] and i < dT_array.size:
                    T = dT_array[i]
                    for attacker in attackers:
                        # calculate ball control probablity for 'attacker' in time interval T+dt
                        PPCF_sum = PPCFatt[i - 1] + PPCFdef[i - 1]
                        dPPCFdT = (1 - PPCF_sum) * attacker.probability_intercept_ball(T) * attacker.lambda_att

                        # make sure it's greater than zero
                        assert dPPCFdT >= 0, "Invalid attacker probability (calculate_pitch_control_at_target)"

                        # total contribution from individual attacker
                        attacker.PPCF += dPPCFdT * params["int_dt"]

                        # add to sum over players in the actor team
                        # (remembering array element is zero at the start of each integration iteration)
                        PPCFatt[i] += attacker.PPCF

                    for defender in defenders:
                        # calculate ball control probablity for 'defender' in time interval T+dt
                        PPCF_sum = PPCFatt[i - 1] + PPCFdef[i - 1]
                        dPPCFdT = (1 - PPCF_sum) * defender.probability_intercept_ball(T) * defender.lambda_def
                        # make sure it's greater than zero
                        assert dPPCFdT >= 0, "Invalid defender probability (calculate_pitch_control_at_target)"

                        # total contribution from individual defender
                        defender.PPCF += dPPCFdT * params["int_dt"]

                        # add to sum over players in the opponent team
                        PPCFdef[i] += defender.PPCF

                    ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
                    i += 1
                if i >= dT_array.size:
                    print(f"Integration failed to converge: {ptot}")

            return PPCFatt[i - 1], PPCFdef[i - 1]

# def generate_pitch_control_for_tracking(
#     tracking_home_df: pd.DataFrame,
#     tracking_away_df: pd.DataFrame,
#     tracking_frame: int,
#     attacking_team: str,
#     params: dict,
#     GK_numbers: list,
#     field_dimen=(
#         106.0,
#         68.0,
#     ),
#     n_grid_cells_x=50,
#     offsides=True,
# ):
#     """generate_pitch_control_for_tracking

#     Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

#     Parameters
#     -----------
#         tracking_home_df: tracking DataFrame for the Home team
#         tracking_away_df: tracking DataFrame for the Away team
#         tracking_frame: tracking frame in type int
#         attacking_team: Home or Away in ball possesion team
#         params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
#         GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
#         field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
#         n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
#                         n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
#         offsides: If True, find and remove offside atacking players from the calculation. Default is True.


#     Returrns
#     -----------
#         PPCF_attack: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
#                Surface for the defending team is just 1-PPCF_attack.
#         xgrid: Positions of the pixels in the x-direction (field length)
#         ygrid: Positions of the pixels in the y-direction (field width)

#     """
#     # get the details of the event (frame, team in possession, ball_start_position)
#     # pass_frame = events.loc[event_id]['Start Frame']
#     # pass_team = events.loc[event_id].Team
#     ball_start_pos = np.array(
#         [
#             tracking_home_df.loc[tracking_frame]["ball_x"],
#             tracking_home_df.loc[tracking_frame]["ball_y"],
#         ]
#     )

#     # break the pitch down into a grid
#     n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
#     dx = field_dimen[0] / n_grid_cells_x
#     dy = field_dimen[1] / n_grid_cells_y
#     xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
#     ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0

#     # initialise pitch control grids for attacking and defending teams
#     PPCF_attack = np.zeros(shape=(len(ygrid), len(xgrid)))
#     PPCF_defense = np.zeros(shape=(len(ygrid), len(xgrid)))

#     # initialise player positions and velocities for pitch control calc
#     # (so that we're not repeating this at each grid cell position)
#     if attacking_team == "Home":
#         attacking_players = initialise_players(tracking_home_df.loc[tracking_frame], "Home", params, GK_numbers[0])
#         defending_players = initialise_players(tracking_away_df.loc[tracking_frame], "Away", params, GK_numbers[1])
#     elif attacking_team == "Away":
#         defending_players = initialise_players(tracking_home_df.loc[tracking_frame], "Home", params, GK_numbers[0])
#         attacking_players = initialise_players(tracking_away_df.loc[tracking_frame], "Away", params, GK_numbers[1])
#     else:
#         assert False, "Team in possession must be either home or away"

#     # find any attacking players that are offside and remove them from the pitch control calculation
#     if offsides:
#         attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)
#     # calculate pitch pitch control model at each location on the pitch
#     for i in range(len(ygrid)):
#         for j in range(len(xgrid)):
#             target_position = np.array([xgrid[j], ygrid[i]])
#             PPCF_attack[i, j], PPCF_defense[i, j] = calculate_pitch_control_at_target(
#                 target_position,
#                 attacking_players,
#                 defending_players,
#                 ball_start_pos,
#                 params,
#             )
#     # check probabilitiy sums within convergence
#     checksum = np.sum(PPCF_attack + PPCF_defense) / float(n_grid_cells_y * n_grid_cells_x)
#     assert 1 - checksum < params["model_converge_tol"], "Checksum failed: %1.3f" % (1 - checksum)
#     return PPCF_attack, xgrid, ygrid, attacking_players
