import os
import pickle
import config

dataset_path = config.DATASET_SHADE_PATH
living_room_test_path = os.path.join(dataset_path, 'livingroom', 'calibrated_livingroom_test.pkl')
with open(living_room_test_path, 'rb') as f:
    living_room_test = pickle.load(f)
print(living_room_test.keys())
