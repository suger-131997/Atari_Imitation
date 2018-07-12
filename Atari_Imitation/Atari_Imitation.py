import os
import time

from PIL import Image
import numpy as np
from tqdm import tqdm

from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D
from keras.optimizers import SGD, Adam

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
EPOCHS = 1

# 



def load_traj_prepro(nb_action, p=0.05):
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

        #生データ用リスト
        traj_list = []
        act_list = []

        # 前処理後用リスト
        status = []
        action = []

        #行数取得
        num_lines = sum(1 for line in open(PATH + '/trajectories/' + GAME_NAME + '/' +traj_num + '.txt'))
        
        # ヘッダ部分を飛ばす(二行分)
        f = open(PATH + '/trajectories/' + GAME_NAME + '/' +traj_num + '.txt')
        next(f)
        next(f)

        # screenとactionをロード
        for img_file, line in tqdm(zip(os.listdir(PATH + '/screens/' + GAME_NAME + "/" + traj_num), f), total=num_lines-2):
            traj_list.append(np.array(Image.open(PATH + '/screens/' + GAME_NAME + "/" + traj_num + "/" +img_file, 'r')))
            act_list.append(int(line.split(",")[4]))
        
        # 状態前処理
        print("Now Preprocess : %s" % traj_num)
        for i in tqdm(range(len(traj_list) // FRAME_SIZE)):
            # 状態を追加
            status.append(preprocess(traj_list[i*FRAME_SIZE:i*FRAME_SIZE+4]))

            # 行動を追加
            act = act_list[i*FRAME_SIZE+3]
                
            #データセットのバグに対応
            j=0
            while act >= nb_action:
                j += 1
                act = act_list[i*FRAME_SIZE+3+j]

            action.append(act)

        # numpy 変換保存
        status_ary.append(np.array(status))
        action_ary.append(np.array(action))

        #メモリ対策
        del traj_list
        del act_list
        del status
        del action

    status = np.empty((0, *INPUT_SHAPE, FRAME_SIZE), 'float32')
    action = np.array([])

    for s, a in tqdm(zip(status_ary, action_ary)):
        status = np.append(status, s, axis=0)
        action = np.append(action, a, axis=0)

    return status, action

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
            
def preprocess(status):
    """状態の前処理"""
    # 状態は4つで1状態
    assert len(status) == FRAME_SIZE

    state = np.empty((*INPUT_SHAPE, FRAME_SIZE), int)

    for i, s in enumerate(status):
        # 配列に追加
        state[:, :, i] = _preprocess(s)

    # 画素値を0～1に正規化
    state = state.astype('float32') / 255.0

    return state.tolist()

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
        status, action = load_traj_prepro(nb_action, 2.0 /667.0)

        # numpyに変換
        status = np.array(status)
        action = np.array(action)

        np.save('data/status.npy', status)
        np.save('data/action.npy', action)
    else:
        # 保存済みデータ使用
        status = np.load('data/status.npy')
        action = np.load('data/action.npy')

    # モデルのコンパイル
    model.compile(optimizer=Adam(),           
                    loss="categorical_crossentropy",                 
                    metrics=["accuracy"])

    # テンソルボード
    tb = TensorBoard(log_dir="./logs")

    # データセットのバグに対応
    ary = np.asarray(action)
    ary = np.where(ary >= nb_action, 0, ary)

    # 学習
    history = model.fit(status, 
                        np_utils.to_categorical(ary, nb_action),                  
                        batch_size=BATCH_SIZE,                       
                        epochs=EPOCHS,                                
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
    env = wrappers.Monitor(env, './movie_folder', video_callable=(lambda ep: True), force=True)
    
    # モデル構築
    model = build_cnn_model(env.action_space.n)

    # モデル学習
    tarin(model, env.action_space.n)

    # テスト
    test(model, env)


if __name__ == "__main__":
    #実行時間計測
    start_time = time.time()
    
    main()

    execution_time = time.time() - start_time
    print(execution_time)