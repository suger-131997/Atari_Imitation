import os

from PIL import Image
import numpy as np
from tqdm import tqdm

from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D
from keras.optimizers import SGD

from keras.utils import np_utils

import gym
from gym import wrappers

# TensorFlowの警告を非表示に
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 軌跡データフォルダ
PATH = "/Users/SHIGAYUYA/Desktop/atari_v2_release"

# 環境名
GAME_NAME = "mspacman"
ENV_NAME = "MsPacman-v0"

# 入力サイズ
INPUT_SHAPE = (84, 84)

# フレーム間隔
FRAME_SIZE = 4

# 学習用定数
BATCH_SIZE = 128
EPOCHS = 100

def load_traj(p=0.05):
    """行動の軌跡の上位pを取得"""
    traj_score = []
    for traj in os.listdir(PATH + '/trajectories/' + GAME_NAME):
        f = open(PATH + '/trajectories/' + GAME_NAME + '/' + traj, 'r')
        end_data = f.readlines()[-1].split(",")
        #[フォルダ名, 最終スコア]
        traj_score.append([os.path.splitext(traj)[0], int(end_data[2])])
        f.close()

    # スコアでソート
    traj_score = sorted(traj_score, key=lambda x:x[1], reverse=True)

    # 軌道をロード
    traj_all = []
    for traj_num, _ in traj_score[:int(len(traj_score)*p)]:
        print("Loading : #{traj_num}")
        traj_sub = []
        
        # ヘッダ部分を飛ばす(二行分)
        f = open(PATH + '/trajectories/' + GAME_NAME + "/" +traj_num + ".txt")
        next(f)
        next(f)

        # screenとactionをロード
        for img_file, line in tqdm(zip(os.listdir(PATH + '/screens/' + GAME_NAME + "/" + traj_num), f)):
            traj_sub.append([np.asarray(Image.open(PATH + '/screens/' + GAME_NAME + "/" + traj_num + "/" +img_file, 'r')), line.split(",")[4]])
        traj_all.append(traj_sub)

    return traj_all
            
def preprocess(status):
    """状態の前処理"""
    # 状態は4つで1状態
    assert len(status) == FRAME_SIZE

    state = np.empty((*INPUT_SHAPE, 0), int)

    for s in status:
        # 画像化
        img = Image.fromarray(s)
        # サイズを入力サイズへ
        img = img.resize(INPUT_SHAPE)
        # グレースケールに
        img = img.convert('L')  
        # 配列に追加
        state = np.append(state, np.array(img), axis=-1)

    # 画素値を0～1に正規化
    state = state.astype('float32') / 255.0

    return state

def build_cnn_model(nb_action):
    """CNNモデル構築"""
    model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", padding="same", input_shape=(*INPUT_SHAPE, FRAME_SIZE)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_action, activation="softmax"))
    return model

def tarin(model):
    """学習"""
    # 軌跡ロード
    raw_trajs = load_traj()

    # 状態前処理
    status = np.empty((0, *INPUT_SHAPE, FRAME_SIZE))
    action = []
    for traj in raw_trajs:
        for i in renge(len(traj) // FRAME_SIZE):
            # 状態を追加
            np.append(status, preprocess(traj[i*FRAME_SIZE:i*FRAME_SIZE+3][0]), axis=0)

            # 行動を追加
            action.append(traj[i*FRAME_SIZE+3][1])

    # モデルのコンパイル
    model.compile(optimizer=SGD(lr=0.01),           
                    loss="categorical_crossentropy",                 
                    metrics=["accuracy"])

    # テンソルボード
    tb = TensorBoard(log_dir="./logs")

    # 学習
    history = model.fit(status, 
                        np_utils.to_categorical(np.asarray(action)),                  
                        batch_size=BATCH_SIZE,                       
                        epochs=EPOCHS,                                
                        callbacks=[tb])

# TO DO
def test(model, env):
    """テスト"""
    while True:
        for i in range(FRAME_SIZE):
            env.render()
            #行動決定
            action = get_action(state)

            #実行
            observation, reward, done, info = env.step(action)

        reward_sum += reward
            
        #終了
        if done == True:
            break
    
def main():
    """メイン関数"""
    # gym環境指定
    env = gym.make(ENV_NAME)

    #動画保存
    env = wrappers.Monitor(env, './movie_folder', video_callable=(lambda ep: True), force=True)
    
    #モデル構築
    model = build_cnn_model(env.action_space.n)

    #モデル学習
    tarin(model)

    #テスト
    #test(model, env)


if __name__ == "__main__":
    load_traj()