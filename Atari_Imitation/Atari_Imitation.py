import os
import time
import argparse

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K

import gym
from gym import wrappers

# TensorFlowの警告を非表示に
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 軌跡データフォルダ
PATH = None

# 環境名
GAME_NAME = "qbert"
ENV_NAME = "QbertNoFrameskip-v4"

# 入力サイズ
INPUT_SHAPE = (84, 84)

# フレーム間隔
FRAME_SIZE = 4

# 学習用定数
BATCH_SIZE = 128
EPOCHS = 15

# 軌道利用割合
USE_TRAJ_RATIO = 0.01

# 前処理実行
RUN_PREPROCESS = False

# ラベルごとの重み適用
USE_CLASS_WEIGHT = False

# Action変換表
"""
Qbertのaction
['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN']
データセットのaction
['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
"""
act_trans_list = (0,1,2,3,4,5,2,2,3,4,2,3,4,5,2,2,3,4)


def load_traj_prepro(nb_action, p=0.01, concat=True):
    """行動の軌跡の上位pを取得し、前処理"""
    traj_score = []
    for traj in os.listdir(PATH + '/trajectories/' + GAME_NAME):
        f = open(PATH + '/trajectories/' + GAME_NAME + '/' + traj, 'r')
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
        df = pd.read_csv(PATH + '/trajectories/' + GAME_NAME + '/' +traj_num + '.txt', skiprows=1)
        traj_list = [np.array(Image.open(PATH + '/screens/' + GAME_NAME + '/' + traj_num + '/' +img_file + '.png', 'r')) for img_file in tqdm(df['frame'].astype('str').values.tolist())]
        act_list = df['action'].astype('int8').values.tolist()

        # 前処理
        print("Now Preprocess : %s" % traj_num)

        status = np.concatenate([preprocess(traj_list[i:i+4], False, False)[np.newaxis, :, :, :] for i in tqdm(range(len(traj_list) // FRAME_SIZE))], axis=0)
        action = [act_trans_list[act_list[i+3]] for i in tqdm(range(len(traj_list) // FRAME_SIZE))]
        

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
        print("Now make batch")
        status = np.concatenate(status_ary, axis=0)
        del status_ary
        action = np.concatenate(action_ary, axis=0)
        del action_ary

        # 状態正規化
        status = status.astype('float32') / 255.0
        print("End make batch")
    else:
        status = [s.astype('float32') / 255.0 for s in status_ary]
        del status_ary
        action = action_ary
        del action_ary

    return status, action

def preprocess(status, tof=True, tol=True):
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
    assert len(status) == FRAME_SIZE

    state = np.empty((*INPUT_SHAPE, FRAME_SIZE), 'int8')

    for i, s in enumerate(status):
        # 配列に追加
        state[:, :, i] = _preprocess(s)

    if tof:    
        # 画素値を0～1に正規化
        state = state.astype('float32') / 255.0

    if tol:
        state = state.tolist()

    return state

def build_cnn_model(nb_action):
    """CNNモデル構築"""
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", padding="same", input_shape=(*INPUT_SHAPE, FRAME_SIZE)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_action, activation="softmax"))

    return model

def tarin(model, nb_action, preprocess=True):
    """学習"""
    status = None
    action = None

    if preprocess:
        # 前処理済み軌跡ロード
        status, action = load_traj_prepro(nb_action, USE_TRAJ_RATIO)

        print('-----------------------------------------------------')
        # 軌跡データ保存
        dirname = 'data_' + str(USE_TRAJ_RATIO)

        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        print('Now Save Numpy')
        np.save(dirname + '/status.npy', status)
        np.save(dirname + '/action.npy', action)
        print('End Save Numpy')
    else:
        # 保存済みデータ使用
        dirname = 'data_' + str(USE_TRAJ_RATIO)

        print('Now Load Numpy')
        status = np.load(dirname + '/status.npy')
        action = np.load(dirname + '/action.npy')
        print('End Load Numpy')

    # モデルのコンパイル
    model.compile(optimizer=Adam(),           
                    loss="categorical_crossentropy",                 
                    metrics=["accuracy"])

    # テンソルボード
    tb = TensorBoard(log_dir="./logs")

    # ラベルの重み付け
    weight = None
    if USE_CLASS_WEIGHT:
        unique, count = np.unique(action, return_counts=True)
        weight = np.max(count) / count
        weight = dict(zip(unique, weight))

    # 学習
    history = model.fit(status, 
                        np_utils.to_categorical(action, nb_action),
                        batch_size=BATCH_SIZE,                       
                        epochs=EPOCHS,
                        class_weight=weight,
                        callbacks=[tb])

def test(model, env):
    """テスト"""
    
    action = 0
    
    #環境初期化
    observation = env.reset()
    while True:
        state = []

        # 初期 or 前回の状態を追加
        state.append(observation)

        for i in range(1, FRAME_SIZE):
            # 描画
            env.render()

            # 行動スキップ
            observation, _, done, _ = env.step(0)

            # 終了
            if done == True:
                break

            # 配列に追加
            state.append(observation)

        # 終了
        if done == True:
            break

        # 状態前処理
        state = preprocess(state)

        # 行動選択
        action = model.predict_on_batch(np.array([state]))

        # 行動
        observation, _, done, _ = env.step(np.argmax(action))
            
        #終了
        if done == True:
            break
    
def main():
    """メイン関数"""
    # gym環境指定
    env = gym.make(ENV_NAME)

    # 動画保存
    env = wrappers.Monitor(env, './movie_folder_' + str(USE_TRAJ_RATIO) + '_' + str(EPOCHS) +'_' + str(BATCH_SIZE) + ('_CW' if USE_CLASS_WEIGHT else ''), video_callable=(lambda ep: True), force=True)
    
    # モデル構築
    model = build_cnn_model(env.action_space.n)

    # モデル学習
    tarin(model, env.action_space.n, RUN_PREPROCESS)

    # テスト
    test(model, env)

    K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('-b', '--batchsize', type=int, default=128)
    parser.add_argument('-p', '--preprocess', action="store_true")
    parser.add_argument('-cw', '--classweight', action="store_true")
    parser.add_argument('--raito', type=float, default=0.01)
    args = parser.parse_args()

    PATH = args.path
    EPOCHS = args.epoch
    BATCH_SIZE = args.batchsize
    RUN_PREPROCESS = args.preprocess
    USE_CLASS_WEIGHT = args.classweight
    USE_TRAJ_RATIO = args.raito

    #実行時間計測
    start_time = time.time()
    
    main()

    execution_time = time.time() - start_time
    print(execution_time)