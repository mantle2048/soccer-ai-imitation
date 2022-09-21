###################################
#  extrace single player state    #
#  player_state:                  #
#       idx: 8 dim                #
#       pos: 2 dim (convert2left) #
#       direction: 2 dim          #
#       speed: 1 dim              #
#       tired: 1 dim              #
#       game_mode: 7 dim          #
#       sticky_action: 10 dim     #
#       ball_owned: 1 dim         #
#       ball_owned_by_us: 1 dim   #
#       ball_owned_by_me: 1 dim   #
#       offside: 1 dim            #
#       total: 35 dim             #
###################################
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
        if obs_temp['ball_owned_team'] == 1:
            obs_reverse['ball_owned_team'] = 0
        elif obs_temp['ball_owned_team'] == 0:
            obs_reverse['ball_owned_team'] = 1
        else:
            obs_reverse['ball_owned_team'] = -1
        obs_reverse['ball_owned_player'] = obs_temp['ball_owned_player']
        obs_reverse['steps_left'] = obs_temp['steps_left']

        return obs_reverse

    def encoder(self, mode, *args,**kwargs):
        if mode == 'complex':
            return self.encoder_complex(*args, **kwargs)
        elif mode == 'simple':
            return self.encoder_simple(*args, **kwargs)

    def encoder_simple(self, obs: Dict, idx: int):
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

    def encoder_complex(self, obs: Dict, idx: int):
        idx_onehot = np.zeros(8)
        idx_onehot[idx] = 1

        if idx < 4:
            player_num = idx + 1
        elif 4 <= idx < 8:
            player_num = idx - 4 + 1

        # pos
        player_pos = obs["left_team"][player_num]
        others_pos = np.concatenate(
            [np.delete(obs["left_team"], player_num, axis=0), obs['right_team']],
            axis=0)
        relative_pos = others_pos - player_pos
        pos_state = np.concatenate([player_pos[None, :], others_pos], axis=0)

        # dir
        player_dir = np.array(obs["left_team_direction"][player_num])
        others_dir = np.concatenate(
            [np.delete(obs["left_team_direction"], player_num, axis=0), obs['right_team_direction']],
            axis=0)
        relative_dir = others_dir - player_dir
        dir_state = np.concatenate([player_dir[None, :], others_dir], axis=0)

        # player_state
        player_speed = np.linalg.norm(player_dir)
        player_tired = obs["left_team_tired_factor"][player_num]
        left_offside, right_offside = self.get_offside(obs)
        player_offside = left_offside[player_num]

        # sticky action
        player_sticky_act = obs["left_agent_sticky_actions"][player_num - 1]

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

        # steps_left = obs['steps_left'] / 3000  # steps left till end

        # score_ratio = 1.0 * (obs['score'][0] - obs['score'][1])
        # score_ratio /= 5.0
        # score_ratio = min(score_ratio, 1.0)
        # score_ratio = max(-1.0, score_ratio)

        # ball_x, ball_y, ball_z = obs["ball"]
        # ball_x_relative = ball_x - player_pos_x
        # ball_y_relative = ball_y - player_pos_y
        # ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        # ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        # ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])

        # ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        state_dict = {
            'idx': idx_onehot,
            'pos': pos_state,
            'direction': dir_state * 100,
            'speed': [player_speed * 100],
            'tired': [player_tired],
            'offside': [player_offside],
            'game_mode': game_mode_onehot,
            'sticky_action': player_sticky_act,
            'ball_state': ball_state,
        }
        return state_dict, {}

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
                if need_reverse:
                    bilibili_obs_dict = feature_encoder.right_to_left(obs_dict)
                    bilibili_act = action_right_to_left(act_list[4:])
                    opponent_obs_dict = obs_dict.copy()
                    opponent_act = act_list[:4]
                else:
                    bilibili_obs_dict = obs_dict.copy()
                    bilibili_act = act_list[:4]
                    opponent_obs_dict = feature_encoder.right_to_left(obs_dict)
                    opponent_act = action_right_to_left(act_list[4:]) 

                act_list = bilibili_act + opponent_act
                obs_list = []
                for i in range(4):
                    player_obs_dict = \
                        feature_encoder.encoder('simple', bilibili_obs_dict, i)[0]
                    player_obs = \
                        concate_observation_from_raw_no_sort(player_obs_dict)
                    obs_list.append(player_obs)
                for i in range(4, 8):
                    player_obs_dict = \
                        feature_encoder.encoder('simple', opponent_obs_dict, i)[0]
                    player_obs = \
                        concate_observation_from_raw_no_sort(player_obs_dict)
                    obs_list.append(player_obs)
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
