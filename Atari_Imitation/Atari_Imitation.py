import os

from PIL import Image
import numpy as np

#軌跡データフォルダ
PATH = "/Users/SHIGAYUYA/Desktop/atari_v2_release"

#環境名
GAME_NAME = "mspacman"
ENV_NAME = "MsPacman-v0"

#入力サイズ
INPUT_SHAPE = (84, 84)

def load_traj(p=0.05):
    """行動の軌跡の上位pを取得"""
    traj_score = []
    for traj in os.listdir(PATH + '/trajectories/' + GAME_NAME):
        f = open(PATH + '/trajectories/' + GAME_NAME + '/' + traj, 'r')
        end_data = f.readlines()[-1].split(",")
        #[フォルダ名, 最終スコア]
        traj_score.append([os.path.splitext(traj)[0], int(end_data[2])])
        f.close()

    #スコアでソート
    traj_score = sorted(traj_score, key=lambda x:x[1], reverse=True)

    #軌道をロード
    traj_all = []
    for traj_num, _ in traj_score[:int(len(traj_score)*0.05)]:
        print(traj_num)
        traj_sub = []
        #ヘッダ部分を飛ばす(二行分)
        f = open(PATH + '/trajectories/' + GAME_NAME + "/" +traj_num + ".txt")
        next(f)
        next(f)
        #screenとactionをロード
        for img_file, line in zip(os.listdir(PATH + '/screens/' + GAME_NAME + "/" + traj_num), f):
            traj_sub.append([Image.open(PATH + '/screens/' + GAME_NAME + "/" + traj_num + "/" +img_file, 'r'), line.split(",")[4]])
        traj_all.append(traj_sub)

        
            
def preprocess(status):
    """状態の前処理"""
    #状態は4つで1状態
    assert len(status) == 4

    state = np.empty((*INPUT_SHAPE, 0), int)

    for s in status:
        #画像化
        img = Image.fromarray(s)
        #サイズを入力サイズへ
        img = img.resize(INPUT_SHAPE)
        #グレースケールに
        img = img.convert('L')  
        #配列に追加
        state = np.append(state, np.array(img), axis=-1)

    #画素値を0～1に正規化
    state = state.astype('float32') / 255.0

    return state

if __name__ == "__main__":
    load_traj()