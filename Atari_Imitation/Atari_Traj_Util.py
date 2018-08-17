import os

from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

# 環境名
GAME_NAME = "qbert"
ENV_NAME = "QbertNoFrameskip-v4"

# 入力サイズ
INPUT_SHAPE = (84, 84)

# フレーム間隔
FRAME_SIZE = 4

# Action変換表
"""
Qbertのaction
['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN']
データセットのaction
['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
"""
act_trans_list = (0,1,2,3,4,5,2,2,3,4,2,3,4,5,2,2,3,4)

def load_traj_prepro(path, nb_action, p=0.01, concat=True, frame_size=FRAME_SIZE):
    """行動の軌跡の上位pを取得し、前処理"""
    traj_score = []
    for traj in os.listdir(path + '/trajectories/' + GAME_NAME):
        f = open(path + '/trajectories/' + GAME_NAME + '/' + traj, 'r')
        end_data = f.readlines()[-1].split(",")
        #[フォルダ名, 最終スコア]
        traj_score.append([os.path.splitext(traj)[0], int(end_data[2])])
        f.close()

    # スコアでソート
    traj_score = sorted(traj_score, key=lambda x:x[1], reverse=True)

    # 軌道ごとに記録する配列
    status_ary = []
    action_ary = []
    for traj_num, _ in traj_score[:int(len(traj_score)*p)]:
        print("Now Loading : %s" % traj_num)
        # データロード
        df = pd.read_csv(path + '/trajectories/' + GAME_NAME + '/' +traj_num + '.txt', skiprows=1)
        traj_list = [np.array(Image.open(path + '/screens/' + GAME_NAME + '/' + traj_num + '/' +img_file + '.png', 'r')) 
                     for img_file in tqdm(df['frame'].astype('str').values.tolist())]
        act_list = df['action'].astype('int8').values.tolist()

        # 前処理
        print("Now Preprocess : %s" % traj_num)

        status = np.concatenate([preprocess(traj_list[i:i+frame_size], frame_size, False, False)[np.newaxis, :, :, :] 
                                 for i in tqdm(range(len(traj_list) // frame_size))], axis=0)
        action = [act_trans_list[act_list[i+(frame_size-1)]] for i in tqdm(range(len(traj_list) // frame_size))]
        

        status_ary.append(status)
        action_ary.append(np.array(action))

        #メモリ対策
        del traj_list
        del act_list
        del status
        del action

    status = None
    action = None

    if concat:
        # numpy展開
        print("Now Concatenate")
        status = np.concatenate(status_ary, axis=0)
        del status_ary
        action = np.concatenate(action_ary, axis=0)
        del action_ary

        # 状態正規化
        status = status.astype('float32') / 255.0
        print("End Concatenate")
    else:
        status = [s.astype('float32') / 255.0 for s in status_ary]
        del status_ary
        action = action_ary
        del action_ary

    return status, action

def preprocess(status, frame_size, tof=True, tol=True):
    """状態の前処理"""

    def _preprocess(observation):
        """画像への前処理"""
        # 画像化
        img = Image.fromarray(observation)
        # サイズを入力サイズへ
        img = img.resize(INPUT_SHAPE)
        # グレースケールに
        img = img.convert('L') 
        # 配列に追加
        return np.array(img)

    # 状態は4つで1状態
    assert len(status) == frame_size

    state = np.empty((*INPUT_SHAPE, frame_size), 'int8')

    for i, s in enumerate(status):
        # 配列に追加
        state[:, :, i] = _preprocess(s)

    if tof:    
        # 画素値を0～1に正規化
        state = state.astype('float32') / 255.0

    if tol:
        state = state.tolist()

    return state
