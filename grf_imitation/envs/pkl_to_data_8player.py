import copy
import os
import os.path as osp
import numpy as np
import _pickle as cPickle
import glob

from tqdm import tqdm
from typing import Dict

action_num = {
    'idle': 0,
    'left': 1,
    'top_left': 2,
    'top': 3,
    'top_right': 4,
    'right': 5,
    'bottom_right': 6,
    'bottom': 7,
    'bottom_left': 8,
    'long_pass': 9,
    'high_pass': 10,
    'short_pass': 11,
    'shot': 12,
    'sprint': 13,
    'release_direction': 14,
    'release_sprint': 15,
    'sliding': 16,
    'dribble': 17,
    'release_dribble': 18,
}
def get_reversed_sticky_actions(actions):
    for act in actions:
        if act[0] == 1:
            act[0] = 0
            act[4] = 1
        elif act[1] == 1:
            act[1] = 0
            act[5] = 1
        elif act[2] == 1:
            act[2] = 0
            act[6] = 1
        elif act[3] == 1:
            act[3] = 0
            act[7] = 1
        elif act[4] == 1:
            act[4] = 0
            act[0] = 1
        elif act[5] == 1:
            act[5] = 0
            act[1] = 1
        elif act[6] == 1:
            act[6] = 0
            act[2] = 1
        elif act[7] == 1:
            act[7] = 0
            act[3] = 1
    return actions


def action_right_to_left(action_list):
    action_reserved = []
    for act in action_list:
        if act == 1:
            action_reserved.append(5)
        elif act == 2:
            action_reserved.append(6)
        elif act == 3:
            action_reserved.append(7)
        elif act == 4:
            action_reserved.append(8)
        elif act == 5:
            action_reserved.append(1)
        elif act == 6:
            action_reserved.append(2)
        elif act == 7:
            action_reserved.append(3)
        elif act == 8:
            action_reserved.append(4)
        else:
            action_reserved.append(act)

    return action_reserved


def obs_enhance(obs):
    obs_reversed = copy.deepcopy(obs)
    reversed_matrix_1 = np.array([1, -1])
    reversed_matrix_2 = np.array([1, -1, 1])

    obs_reversed["ball"] = obs["ball"] * reversed_matrix_2
    obs_reversed["ball_direction"] = obs["ball_direction"] * reversed_matrix_2
    obs_reversed["ball_rotation"] = obs["ball_rotation"] * reversed_matrix_2

    obs_reversed["left_team"] = obs["left_team"] * reversed_matrix_1
    obs_reversed["left_team_direction"] = obs["left_team_direction"] * reversed_matrix_1

    obs_reversed["right_team"] = obs["right_team"] * reversed_matrix_1
    obs_reversed["right_team_direction"] = obs["right_team_direction"] * reversed_matrix_1

    for i in range(len(obs["left_agent_sticky_actions"])):
        if obs["left_agent_sticky_actions"][i][1] == 1:
            obs_reversed["left_agent_sticky_actions"][i][1] = 0
            obs_reversed["left_agent_sticky_actions"][i][7] = 1
        elif obs["left_agent_sticky_actions"][i][3] == 1:
            obs_reversed["left_agent_sticky_actions"][i][3] = 0
            obs_reversed["left_agent_sticky_actions"][i][5] = 1
        elif obs["left_agent_sticky_actions"][i][5] == 1:
            obs_reversed["left_agent_sticky_actions"][i][5] = 0
            obs_reversed["left_agent_sticky_actions"][i][3] = 1
        elif obs["left_agent_sticky_actions"][i][7] == 1:
            obs_reversed["left_agent_sticky_actions"][i][7] = 0
            obs_reversed["left_agent_sticky_actions"][i][1] = 1

    for i in range(len(obs["right_agent_sticky_actions"])):
        if obs["right_agent_sticky_actions"][i][1] == 1:
            obs_reversed["right_agent_sticky_actions"][i][1] = 0
            obs_reversed["right_agent_sticky_actions"][i][7] = 1
        elif obs["right_agent_sticky_actions"][i][3] == 1:
            obs_reversed["right_agent_sticky_actions"][i][3] = 0
            obs_reversed["right_agent_sticky_actions"][i][5] = 1
        elif obs["right_agent_sticky_actions"][i][5] == 1:
            obs_reversed["right_agent_sticky_actions"][i][5] = 0
            obs_reversed["right_agent_sticky_actions"][i][3] = 1
        elif obs["right_agent_sticky_actions"][i][7] == 1:
            obs_reversed["right_agent_sticky_actions"][i][7] = 0
            obs_reversed["right_agent_sticky_actions"][i][1] = 1

    return obs_reversed


def get_enhanced_action(action_list):
    agent_action_enhanced = []
    for act in action_list:
        if act == 2:
            agent_action_enhanced.append(8)
        elif act == 4:
            agent_action_enhanced.append(6)
        elif act == 6:
            agent_action_enhanced.append(4)
        elif act == 8:
            agent_action_enhanced.append(2)
        else:
            agent_action_enhanced.append(act)
    return agent_action_enhanced


def concate_observation_from_raw_no_sort(obs):
    obs_cat = np.hstack([np.array(obs[k], dtype=np.float32).flatten() for k in obs])
    return obs_cat


def get_action_num(action):
    action_list = []
    for act in action:
        action_list.append(action_num[str(act)])
    return action_list


class MyFeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0
        self.last_loffside = np.zeros(5, np.float32)
        self.last_roffside = np.zeros(5, np.float32)

    def right_to_left(self, obs):
        obs_reverse = {}
        obs_temp = copy.deepcopy(obs)
        obs_reverse['ball'] = obs_temp['ball'] * np.array((-1, -1, 1))
        obs_reverse['ball_direction'] = obs_temp['ball_direction'] * np.array((-1, -1, 1))
        obs_reverse['ball_rotation'] = obs_temp['ball_rotation'] * np.array((-1, -1, 1))

        obs_reverse['right_team'] = obs_temp['left_team'] * np.array((-1, -1))
        obs_reverse['right_team_direction'] = obs_temp['left_team_direction'] * np.array((-1, -1))
        obs_reverse['right_team_tired_factor'] = obs_temp['left_team_tired_factor']
        obs_reverse['right_team_active'] = obs_temp['left_team_active']
        obs_reverse['right_team_yellow_card'] = obs_temp['left_team_yellow_card']
        obs_reverse['right_team_roles'] = obs_temp['left_team_roles']
        obs_reverse['right_team_designated_player'] = obs_temp['left_team_designated_player']

        obs_reverse['left_team'] = obs_temp['right_team'] * np.array((-1, -1))
        obs_reverse['left_team_direction'] = obs_temp['right_team_direction'] * np.array((-1, -1))
        obs_reverse['left_team_tired_factor'] = obs_temp['right_team_tired_factor']
        obs_reverse['left_team_active'] = obs_temp['right_team_active']
        obs_reverse['left_team_yellow_card'] = obs_temp['right_team_yellow_card']
        obs_reverse['left_team_roles'] = obs_temp['right_team_roles']
        obs_reverse['left_team_designated_player'] = obs_temp['right_team_designated_player']

        obs_reverse['right_agent_sticky_actions'] = get_reversed_sticky_actions(
            obs_temp['left_agent_sticky_actions'])
        obs_reverse['right_agent_controlled_player'] = obs_temp['left_agent_controlled_player']

        obs_reverse['left_agent_sticky_actions'] = get_reversed_sticky_actions(
            obs_temp['right_agent_sticky_actions'])
        obs_reverse['left_agent_controlled_player'] = obs_temp['right_agent_controlled_player']

        obs_reverse['game_mode'] = obs_temp['game_mode']
        obs_reverse['score'] = list(reversed(obs_temp['score']))
        obs_reverse['ball_owned_team'] = obs_temp['ball_owned_team']
        obs_reverse['ball_owned_player'] = obs_temp['ball_owned_player']
        obs_reverse['steps_left'] = obs_temp['steps_left']

        return obs_reverse

    def encoder_8player(self, obs: Dict) -> np.ndarray:
        l_pos = obs['left_team'][1:]
        r_pos = obs['right_team'][1:]
        l_dir = obs['left_team_direction'][1:] * 100
        r_dir = obs['right_team_direction'][1:] * 100
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction'].copy()
        ball_dir[0] *= 20
        ball_dir[1] *= 20
        ball_dir[2] *= 5
        obs_dict = {
            'l_pos': l_pos,
            'r_pos': r_pos,
            'l_dir': l_dir,
            'r_dir': r_dir,
            'ball_pos': ball_pos,
            'ball_dir': ball_dir
        }
        obs_new = concate_observation_from_raw_no_sort(obs_dict)
        return obs_new

    def encoder_left(self, obs, player_index):

        player_num = player_index + 1

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["left_agent_sticky_actions"][player_num - 1][9]
        is_sprinting = obs["left_agent_sticky_actions"][player_num - 1][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail_new(obs, player_num, ball_distance)

        player_state = np.concatenate(
            (
                # avail[2:],
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        player_history_state = np.concatenate(
            (
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array([obs["ball_direction"][0] * 20, obs["ball_direction"][1] * 20, obs["ball_direction"][2] * 5]),
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )
        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_relatvie = obs_left_team - obs["left_team"][player_num]
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["left_team"][player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(obs["left_team_tired_factor"], player_num, axis=0).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        left_team_history_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
            ),
            axis=1,
        )

        obs_right_team = np.array(obs["right_team"])
        obs_right_relative = obs_right_team - obs["left_team"][player_num]
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        right_team_history_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
            ),
            axis=1,
        )

        steps_left = obs['steps_left']  # steps left till end
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501  # steps left till halfend
        half_steps_left = 1.0 * min(half_steps_left, 300.0)  # clip
        half_steps_left /= 300.0

        score_ratio = 1.0 * (obs['score'][0] - obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)

        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs['game_mode']] = 1
        match_state = np.concatenate(
            (
                np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]),
                game_mode
            )
        )

        # offside
        l_o, r_o = self.get_offside(obs)
        offside = np.concatenate(
            (
                l_o,
                r_o
            )
        )

        # card
        card = np.concatenate(
            (
                obs['left_team_yellow_card'],
                obs['left_team_active'],
                obs['right_team_yellow_card'],
                obs['right_team_active']
            )
        )

        # sticky_action
        sticky_action = obs["left_agent_sticky_actions"][player_num - 1]

        # ball_distance
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["ball"][:2], axis=1, keepdims=False
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["ball"][:2], axis=1, keepdims=False
        )
        ball_distance = np.concatenate(
            (
                left_team_distance,
                right_team_distance
            )
        )
        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
            "avail": avail,
            "match_state": match_state,
            "offside": offside,
            "card": card,
            "sticky_action": sticky_action,
            "ball_distance": ball_distance
        }

        history_state_dict = {
            "player": player_history_state,
            "ball": ball_state,
            "left_team": left_team_history_state,
            "right_team": right_team_history_state,
            "offside": offside,
            "ball_distance": ball_distance
        }

        return state_dict, history_state_dict

    def encoder_right(self, obs, player_index):
        player_num = player_index + 1

        player_pos_x, player_pos_y = obs["right_team"][player_num]
        player_direction = np.array(obs["right_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["right_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["right_team_tired_factor"][player_num]
        is_dribbling = obs["right_agent_sticky_actions"][player_num - 1][9]
        is_sprinting = obs["right_agent_sticky_actions"][player_num - 1][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 1:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 0:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail_new(obs, player_num, ball_distance)

        player_state = np.concatenate(
            (
                # avail[2:],
                obs["right_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        player_history_state = np.concatenate(
            (
                obs["right_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array([obs["ball_direction"][0] * 20, obs["ball_direction"][1] * 20, obs["ball_direction"][2] * 5]),
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )
        obs_right_team = np.delete(obs["right_team"], player_num, axis=0)
        obs_right_relatvie = obs_right_team - obs["right_team"][player_num]
        obs_right_team_direction = np.delete(
            obs["right_team_direction"], player_num, axis=0
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["right_team"][player_num], axis=1, keepdims=True)
        right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
        right_team_tired = np.delete(obs["right_team_tired_factor"], player_num, axis=0).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relatvie * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        right_team_history_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relatvie * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
            ),
            axis=1,
        )

        obs_left_team = np.array(obs["left_team"])
        obs_left_relative = obs_left_team - obs["right_team"][player_num]
        obs_left_team_direction = np.array(obs["left_team_direction"])
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["right_team"][player_num], axis=1, keepdims=True
        )
        left_team_speed = np.linalg.norm(
            obs_left_team_direction, axis=1, keepdims=True
        )
        left_team_tired = np.array(obs["left_team_tired_factor"]).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relative * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        left_team_history_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relative * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
            ),
            axis=1,
        )

        steps_left = obs['steps_left']  # steps left till end
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501  # steps left till halfend
        half_steps_left = 1.0 * min(half_steps_left, 300.0)  # clip
        half_steps_left /= 300.0

        score_ratio = 1.0 * (obs['score'][0] - obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)

        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs['game_mode']] = 1
        match_state = np.concatenate(
            (
                np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]),
                game_mode
            )
        )

        # offside
        l_o, r_o = self.get_offside(obs)
        offside = np.concatenate(
            (
                l_o,
                r_o
            )
        )

        # card
        card = np.concatenate(
            (
                obs['left_team_yellow_card'],
                obs['left_team_active'],
                obs['right_team_yellow_card'],
                obs['right_team_active']
            )
        )

        # sticky_action
        sticky_action = obs["right_agent_sticky_actions"][player_num - 1]

        # ball_distance
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["ball"][:2], axis=1, keepdims=False
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["ball"][:2], axis=1, keepdims=False
        )
        ball_distance = np.concatenate(
            (
                left_team_distance,
                right_team_distance
            )
        )
        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
            "avail": avail,
            "match_state": match_state,
            "offside": offside,
            "card": card,
            "sticky_action": sticky_action,
            "ball_distance": ball_distance
        }

        history_state_dict = {
            "player": player_history_state,
            "ball": ball_state,
            "left_team": left_team_history_state,
            "right_team": right_team_history_state,
            "offside": offside,
            "ball_distance": ball_distance
        }

        return state_dict, history_state_dict

    def encode(self, obs, player_index):
        player_num = player_index + 1

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["left_agent_sticky_actions"][player_num - 1][9]
        is_sprinting = obs["left_agent_sticky_actions"][player_num - 1][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail_new(obs, player_num, ball_distance)

        player_state = np.concatenate(
            (
                # avail[2:],
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        player_history_state = np.concatenate(
            (
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array([obs["ball_direction"][0] * 20, obs["ball_direction"][1] * 20, obs["ball_direction"][2] * 5]),
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )
        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_relatvie = obs_left_team - obs["left_team"][player_num]
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["left_team"][player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(obs["left_team_tired_factor"], player_num, axis=0).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        left_team_history_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
            ),
            axis=1,
        )

        obs_right_team = np.array(obs["right_team"])
        obs_right_relative = obs_right_team - obs["left_team"][player_num]
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        right_team_history_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
            ),
            axis=1,
        )

        steps_left = obs['steps_left']  # steps left till end
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501  # steps left till halfend
        half_steps_left = 1.0 * min(half_steps_left, 300.0)  # clip
        half_steps_left /= 300.0

        score_ratio = 1.0 * (obs['score'][0] - obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)

        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs['game_mode']] = 1
        match_state = np.concatenate(
            (
                np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]),
                game_mode
            )
        )

        # offside
        l_o, r_o = self.get_offside(obs)
        offside = np.concatenate(
            (
                l_o,
                r_o
            )
        )

        # card
        card = np.concatenate(
            (
                obs['left_team_yellow_card'],
                obs['left_team_active'],
                obs['right_team_yellow_card'],
                obs['right_team_active']
            )
        )

        # sticky_action
        sticky_action = obs["left_agent_sticky_actions"][player_num - 1]

        # ball_distance
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["ball"][:2], axis=1, keepdims=False
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["ball"][:2], axis=1, keepdims=False
        )
        ball_distance = np.concatenate(
            (
                left_team_distance,
                right_team_distance
            )
        )
        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
            "avail": avail,
            "match_state": match_state,
            "offside": offside,
            "card": card,
            "sticky_action": sticky_action,
            "ball_distance": ball_distance
        }

        history_state_dict = {
            "player": player_history_state,
            "ball": ball_state,
            "left_team": left_team_history_state,
            "right_team": right_team_history_state,
            "offside": offside,
            "ball_distance": ball_distance
        }

        return state_dict, history_state_dict

    def get_offside(self, obs):
        ball = np.array(obs['ball'][:2])
        ally = np.array(obs['left_team'])
        enemy = np.array(obs['right_team'])

        # 任意球、角球等没有越位，只有正常比赛有越位
        if obs['game_mode'] != 0:
            self.last_loffside = np.zeros(5, np.float32)
            self.last_roffside = np.zeros(5, np.float32)
            return np.zeros(5, np.float32), np.zeros(5, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        # 当一方控球时才判断是否越位
        if obs['ball_owned_team'] > -1:
            effective_ownball_team = obs['ball_owned_team']
            effective_ownball_player = obs['ball_owned_player']
            need_recalc = True
        else:
            # 没有控球但是离球很近也要判断越位
            # 有这种情况比如一脚传球时obs['ball_owned_team'] 时不会显示的
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            # 我方控球
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            # 对方控球
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(5, np.float32)
        right_offside = np.zeros(5, np.float32)

        if effective_ownball_team == 0:
            # 所有对方球员的x坐标加入排序
            # 取倒数第二名防守球员作为越位线
            right_xs = [obs['right_team'][k][0] for k in range(0, 5)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            # 将倒数第二名防守球员的位置和球比较，更深的成为越位线
            offside_line = max(right_xs[-2], ball[0])

            # 己方守门员不参与进攻，不为其计算越位标志，直接用0的初始化
            # 己方半场不计算越位
            for k in range(1, 5):
                if obs['left_team'][k][0] > offside_line and k != effective_ownball_player \
                        and obs['left_team'][k][0] > 0.0:
                    left_offside[k] = 1.0
        else:
            left_xs = [obs['left_team'][k][0] for k in range(0, 5)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            # 左右半场左边相反
            offside_line = min(left_xs[1], ball[0])

            # 左右半场左边相反
            for k in range(1, 5):
                if obs['right_team'][k][0] < offside_line and k != effective_ownball_player \
                        and obs['right_team'][k][0] < 0.0:
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside

    def _get_avail_new(self, obs, player_num, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.03:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["left_agent_sticky_actions"][player_num - 1]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
                -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)


def main(pkl_path, target_side='winner', use_data_enhance=False):
    if use_data_enhance:
        npz_suffix = '_'.join(['npz', 'aug'])
    else:
        npz_suffix = 'npz'
    npz_path = pkl_path.replace('pkl', npz_suffix)
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    file_path = [os.path.join(pkl_path, f) for f in os.listdir(pkl_path)]
    file_num = len(file_path)
    print('Has total %d files ' % file_num)
    for i in tqdm(range(file_num)):
        file = file_path[i]
        print('Processing the %d/%d pkl file: %s' % (i + 1, file_num, file))

        with open(file, 'rb+') as f:
            info = cPickle.load(f)

        if 3000 in info.keys():
            last_frame = info[3000]
            score = last_frame['observation']['score']
            if score[0] > score[1]:
                winner = 'left'
                loser = 'right'
            elif score[1] > score[0]:
                winner = 'right'
                loser = 'left'
            else:
                print('file %s is the tie match which is not used ...'.format(file))
                continue
        else:
            print('file %s is incomplete which is not used !'.format(file))
            continue

        if target_side == 'winner':
            target_ = winner
        elif target_side == 'loser':
            target_ = loser
        else:
            raise 'wrong target side'

        feature_encoder = MyFeatureEncoder()

        one_episode_obs, one_episode_actions = [], []
        if target_ == 'left':
            print('target side is left ...')
            for step in range(3000):
                frame = info[step + 1]
                # get actions
                action = frame['debug']['action'][:4]
                action_list = get_action_num(action)
                # get states
                obs = frame['observation']
                # use data enhancement
                if use_data_enhance:
                    obs_enhanced = obs_enhance(obs)
                    action_list_enhanced = get_enhanced_action(action_list)
                # normal data process
                obs_list = []
                for player_index in range(4):
                    state_dict, history_state_dict = feature_encoder.encode(obs, player_index)
                    state = concate_observation_from_raw_no_sort(state_dict)
                    obs_list.append(state)
                one_episode_obs.extend(obs_list)
                one_episode_actions.extend(action_list)
                # enhanced data process
                if use_data_enhance:
                    obs_enhanced_list = []
                    for player_index in range(4):
                        state_dict, history_state_dict = feature_encoder.encode(obs_enhanced, player_index)
                        state = concate_observation_from_raw_no_sort(state_dict)
                        obs_enhanced_list.append(state)
                    one_episode_obs.extend(obs_enhanced_list)
                    one_episode_actions.extend(action_list_enhanced)

        elif target_ == 'right':
            print('target side is right, need the reverse operation ...')
            for step in range(3000):
                frame = info[step + 1]
                # get actions
                action = frame['debug']['action'][4:]
                action_list = get_action_num(action)
                action_list_reversed = action_right_to_left(action_list)
                # get states
                obs = frame['observation']
                obs_reversed = feature_encoder.right_to_left(obs)

                if use_data_enhance:
                    obs_enhanced = obs_enhance(obs_reversed)
                    action_list_enhanced = get_enhanced_action(action_list_reversed)

                obs_list = []
                for player_index in range(4):
                    state_dict, history_state_dict = feature_encoder.encode(obs_reversed, player_index)
                    state = concate_observation_from_raw_no_sort(state_dict)
                    obs_list.append(state)

                one_episode_obs.extend(obs_list)
                one_episode_actions.extend(action_list_reversed)

                if use_data_enhance:
                    obs_enhanced_list = []
                    for player_index in range(4):
                        state_dict, history_state_dict = feature_encoder.encode(obs_enhanced, player_index)
                        state = concate_observation_from_raw_no_sort(state_dict)
                        obs_enhanced_list.append(state)
                    one_episode_obs.extend(obs_enhanced_list)
                    one_episode_actions.extend(action_list_enhanced)

        one_episode_obs_array = np.array(one_episode_obs)
        one_episode_actions_array = np.array(one_episode_actions)

        file_name = osp.split(file)[-1].replace('pkl', 'npz')
        npz_file = osp.join(npz_path, file_name)
        np.savez(npz_file, data=one_episode_obs_array, label=one_episode_actions_array)
        print('file %s is saved' % npz_file)

def pkl2npz(pkl_dir: Dict):
    pkl_paths = glob.glob(osp.join(pkl_dir, '*.pkl'))
    print(f"{len(pkl_paths)} pkl files to be processed !")
    feature_encoder = MyFeatureEncoder()
    for pkl_path in tqdm(pkl_paths, desc='processing pkl files'):
        obss, acts = [], []
        with open(pkl_path, 'rb') as f:
            frames = cPickle.load(f)
            if 3000 not in frames.keys(): continue
            score_left, score_right = frames[3000]['observation']['score']
            if score_right > score_left: need_reverse = True
            for step, frame in frames.items():
                obs_dict = frame['observation']
                raw_act = frame['debug']['action']
                act_list = get_action_num(raw_act)
                left_act, right_act = act_list[:4], act_list[4:]
                right_act = action_right_to_left(right_act)
                if need_reverse:
                    obs_dict = feature_encoder.right_to_left(obs_dict)
                    act_list = act_list[4:] + act_list[:4]
                else:
                    act_list = act_list[:4] + act_list[4:]
                obs_list = []
                for i in range(4):
                    left_obs_dict = feature_encoder.encoder_left(obs_dict, i)[0]
                    left_obs = concate_observation_from_raw_no_sort(left_obs_dict)
                    right_obs_dict = feature_encoder.encoder_right(obs_dict, i)[0]
                    right_obs = concate_observation_from_raw_no_sort(right_obs_dict)
                    obs_list.append(left_obs)
                    obs_list.append(right_obs)
                # act = np.array(act_list)
                # obss.append(obs)
                # acts.append(act)
                obss.extend(obs_list)
                acts.extend(act_list)
            obss = np.array(obss)
            acts = np.array(acts)
        npz_dir = pkl_dir.replace('pkl', 'npz')
        os.makedirs(npz_dir, exist_ok=True)
        npz_name = osp.split(pkl_path)[-1].replace('pkl', 'npz')
        npz_path = osp.join(npz_dir, npz_name)
        np.savez(npz_path, data=obss, label=acts)


if __name__ == '__main__':
    pkl_dir = osp.join(os.getcwd(), 'data/win_pkl')
    assert 'win' in pkl_dir
    pkl2npz(pkl_dir)
