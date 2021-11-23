

from pathlib import Path

root_path = '/data/appli_PITSI/users/targe/FastML/'

folder_data = 'data/'
folder_train = 'train/'
folder_test = 'test/'

folder_models = 'models/'

if __name__ == '__main__':

    Path(root_path + folder_data + folder_train).mkdir(parents=True, exist_ok=True)
    Path(root_path + folder_data + folder_test).mkdir(parents=True, exist_ok=True)
    Path(root_path + folder_models).mkdir(parents=True, exist_ok=True)
