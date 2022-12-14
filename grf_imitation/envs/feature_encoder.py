import copy
import numpy as np
from typing import Dict


def maxpooling(feature_map, size=2, stride=2):
    player = feature_map.shape[0]
    height = feature_map.shape[1]
    width = feature_map.shape[2]
    channel = feature_map.shape[3]
    padding_height = np.uint16(round((height - size + 1) / stride))
    padding_width = np.uint16(round((width - size + 1) / stride))
    # print(padding_height, padding_width)

    pool_out = np.zeros((player, padding_height, padding_width, channel), dtype=np.float32)

    for player_num in range(player):
        for channel_num in range(channel):
            out_height = 0
            for r in np.arange(0, height, stride):
                out_width = 0
                for c in np.arange(0, width, stride):
                    pool_out[player_num, out_height, out_width, channel_num] = \
                        np.max(feature_map[player_num, r:r + size, c:c + size, channel_num])
                    out_width = out_width + 1
                out_height = out_height + 1
    return pool_out


def obs_enhance(obs):
    obs_reversed = copy.deepcopy(obs)
    reversed_matrix_1 = np.concatenate((np.ones([11, 1]), -np.ones([11, 1])), axis=1)
    reversed_matrix_2 = np.array([1, -1, 1])

    obs_reversed["left_team"] = obs["left_team"] * reversed_matrix_1
    obs_reversed["left_team_direction"] = obs["left_team_direction"] * reversed_matrix_1
    obs_reversed["right_team"] = obs["right_team"] * reversed_matrix_1
    obs_reversed["right_team_direction"] = obs["right_team_direction"] * reversed_matrix_1

    obs_reversed["ball"] = obs["ball"][0] * reversed_matrix_2
    obs_reversed["ball_direction"] = obs["ball_direction"][0] * reversed_matrix_2
    obs_reversed["ball_rotation"] = obs["ball_rotation"][0] * reversed_matrix_2

    if obs["sticky_actions"][1] == 1:
        obs_reversed["sticky_actions"][1] = 0
        obs_reversed["sticky_actions"][7] = 1
    elif obs["sticky_actions"][3] == 1:
        obs_reversed["sticky_actions"][3] = 0
        obs_reversed["sticky_actions"][5] = 1
    elif obs["sticky_actions"][5] == 1:
        obs_reversed["sticky_actions"][5] = 0
        obs_reversed["sticky_actions"][3] = 1
    elif obs["sticky_actions"][7] == 1:
        obs_reversed["sticky_actions"][7] = 0
        obs_reversed["sticky_actions"][1] = 1
    return obs_reversed


def position_encoder(team_position):
    position_one_hot_list = []
    for player_position in team_position:
        position_one_hot = [0] * 12
        pos_x, pos_y = player_position[0], player_position[1]
        if -1 <= pos_x < -0.5 and 0.1367 <= pos_y < 0.41:
            position_one_hot[0] = 1
        elif -1 <= pos_x < -0.5 and -0.1367 <= pos_y < 0.1367:
            position_one_hot[1] = 1
        elif -1 <= pos_x < -0.5 and -0.41 <= pos_y < -0.1367:
            position_one_hot[2] = 1
        elif -0.5 <= pos_x < -1 and 0.1367 <= pos_y < 0.41:
            position_one_hot[3] = 1
        elif -0.5 <= pos_x < -1 and -0.1367 <= pos_y < 0.1367:
            position_one_hot[4] = 1
        elif -0.5 <= pos_x < -1 and -0.41 <= pos_y < -0.1367:
            position_one_hot[5] = 1
        elif 0 <= pos_x < 0.5 and 0.1367 <= pos_y < 0.41:
            position_one_hot[6] = 1
        elif 0 <= pos_x < 0.5 and -0.1367 <= pos_y < 0.1367:
            position_one_hot[7] = 1
        elif 0 <= pos_x < 0.5 and -0.41 <= pos_y < -0.1367:
            position_one_hot[8] = 1
        elif 0.5 <= pos_x < 1 and 0.1367 <= pos_y < 0.41:
            position_one_hot[9] = 1
        elif 0.5 <= pos_x < 1 and -0.1367 <= pos_y < 0.1367:
            position_one_hot[10] = 1
        elif 0.5 <= pos_x < 1 and -0.41 <= pos_y < -0.1367:
            position_one_hot[11] = 1
        position_one_hot_list.append(position_one_hot)
    position_one_hot_array = np.array(position_one_hot_list)
    return position_one_hot_array

class MyFeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0
        self.last_loffside = np.zeros(5, np.float32)
        self.last_roffside = np.zeros(5, np.float32)

    def get_feature_dims(self):
        dims = {
            "player": 19,
            "ball": 18,
            "left_team": 36,
            "left_team_closest": 9,
            "right_team": 45,
            "right_team_closest": 9,
            "avail": 19,
            "match_state": 10,
            "offside": 10,
            "card": 20,
            "sticky_action": 10,
            "ball_distance": 9,
        }
        return dims

    def get_history_feature_dims(self):
        dims = {
            "player": 19,
            "ball": 18,
            "left_team": 32,
            "right_team": 40,
            "offside": 10,
            "ball_distance": 9,
        }
        return dims

    def get_offside(self, obs):
        ball = np.array(obs['ball'][:2])
        ally = np.array(obs['left_team'])
        enemy = np.array(obs['right_team'])

        # ???????????????????????????????????????????????????????????????
        if obs['game_mode'] != 0:
            self.last_loffside = np.zeros(5, np.float32)
            self.last_roffside = np.zeros(5, np.float32)
            return np.zeros(5, np.float32), np.zeros(5, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        # ???????????????????????????????????????
        if obs['ball_owned_team'] > -1:
            effective_ownball_team = obs['ball_owned_team']
            effective_ownball_player = obs['ball_owned_player']
            need_recalc = True
        else:
            # ????????????????????????????????????????????????
            # ????????????????????????????????????obs['ball_owned_team'] ??????????????????
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            # ????????????
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            # ????????????
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
            # ?????????????????????x??????????????????
            # ?????????????????????????????????????????????
            right_xs = [obs['right_team'][k][0] for k in range(0, 5)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            # ??????????????????????????????????????????????????????????????????????????????
            offside_line = max(right_xs[-2], ball[0])

            # ????????????????????????????????????????????????????????????????????????0????????????
            # ???????????????????????????
            for k in range(1, 5):
                if obs['left_team'][k][0] > offside_line and k != effective_ownball_player \
                        and obs['left_team'][k][0] > 0.0:
                    left_offside[k] = 1.0
        else:
            left_xs = [obs['left_team'][k][0] for k in range(0, 5)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            # ????????????????????????
            offside_line = min(left_xs[1], ball[0])

            # ????????????????????????
            for k in range(1, 5):
                if obs['right_team'][k][0] < offside_line and k != effective_ownball_player \
                        and obs['right_team'][k][0] < 0.0:
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside

    def encoder(self, obs, idx):

        idx_onehot = np.zeros(8)
        idx_onehot[idx] = 1

        if idx < 4:
            player_num = idx + 1
        elif 4 <= idx < 8:
            player_num = idx - 4 + 1

        # pos
        player_pos = obs["left_team"][player_num]
        relative2goal_pos = obs['left_team'] - np.array([1.0, 0.0])
        relative2goal_distance = np.linalg.norm(
            relative2goal_pos, axis=1, keepdims=True)
        cloest2goal_num = relative2goal_distance.argmin()

        teammate_pos = (obs['left_team'] - player_pos)[cloest2goal_num]
        
        relative2me_pos = obs['right_team'] - player_pos
        relative2me_distance = np.linalg.norm(
            relative2goal_pos, axis=1, keepdims=True)
        cloest2me_num = relative2me_distance.argmin()

        opponent_pos = (obs['right_team'] - player_pos)[cloest2me_num]
        pos_state = np.concatenate(
            [player_pos, teammate_pos, opponent_pos], axis = 0
        )

        # dir
        player_dir = obs["left_team_direction"][player_num]
        teammate_dir = (obs['left_team_direction'] - player_dir)[cloest2goal_num]
        opponent_dir = (obs['right_team_direction'] - player_dir)[cloest2me_num]
        dir_state = np.concatenate(
            [player_dir, teammate_dir, opponent_dir], axis = 0
        )

        # player_state
        player_speed = np.linalg.norm(player_dir)
        player_tired = obs["left_team_tired_factor"][player_num]
        left_offside, right_offside = self.get_offside(obs)
        player_offside = left_offside[player_num]
        player_state = np.array([player_speed, player_tired, player_offside])

        # sticky action
        # player_sticky_act = obs["sticky_actions"]

        # game mode
        game_mode = obs['game_mode']
        game_mode_onehot = np.zeros(7)
        game_mode_onehot[game_mode] = 1

        # ball state
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0

        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0

        # ball_owned_by_me = 0.0
        # if ball_owned_by_us and obs['ball_owned_player'] == player_num:
        #     ball_owned_by_me = 1.0

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos[0]
        ball_y_relative = ball_y - player_pos[1]
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])

        ball_state = np.concatenate(
            (
                np.array([ball_x_relative, ball_y_relative]),
                np.array([obs["ball_direction"][0] * 20, obs["ball_direction"][1] * 20, obs["ball_direction"][2] * 5]),
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )
        state_dict = {
            'idx': idx_onehot,
            'pos': pos_state,
            'direction': dir_state * 100,
            'player_state': player_state,
            'game_mode': game_mode_onehot,
            'ball_state': ball_state,
        }
        return state_dict, {}

    def _get_avail(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            MOVE,
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
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
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
            ) = (0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
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
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _get_avail_new(self, obs, ball_distance):
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
        sticky_actions = obs["sticky_actions"]
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
        elif (0.64 <= ball_x <= 1.0) and (
                -0.27 <= ball_y <= 0.27
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

