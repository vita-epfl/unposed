import numpy as np

DIM = 3
ALL_JOINTS_COUNT = {'human3.6m': 96, 'amass': 66, '3dpw': 66}
ALL_JOINTS = {'human3.6m': np.arange(0, ALL_JOINTS_COUNT['human3.6m']),
              'amass': np.arange(0, ALL_JOINTS_COUNT['amass']),
              '3dpw': np.arange(0, ALL_JOINTS_COUNT['3dpw'])
              }
JOINTS_TO_INCLUDE = {'human3.6m': np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                           26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                           46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                           75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92]),
                     'amass': np.arange(4 * DIM, 22 * DIM), '3dpw': np.arange(4 * DIM, 22 * DIM)}
INCLUDED_JOINTS_COUNT = {'human3.6m': 66, 'amass': 54, '3dpw': 54}
JOINTS_TO_IGNORE = {'human3.6m': np.array([48, 49, 50, 60, 61, 62, 69, 70, 71, 72, 73, 74, 84, 85, 86, 93, 94, 95])}
JOINTS_EQUAL = {'human3.6m': np.array([39, 40, 41, 57, 58, 59, 66, 67, 68, 39, 40, 41, 81, 82, 83, 90, 91, 92])}
H36M_SUBJECTS = np.array([[1, 6, 7, 8, 9], [11], [5]], dtype='object')
H36_ACTIONS = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
SCALE_RATIO = {'human3.6m': 0.001, 'amass': 1, '3dpw': 1}
SKIP_RATE = {'human3.6m': 1, 'amass': 5, '3dpw': 5}
MPJPE_COEFFICIENT = {'human3.6m': 1, 'amass': 1000, '3dpw': 1000}  # Multiplied to the mpjpe loss to convert them to
# millimeters
TRAIN_K, VALID_K, TEST_K = 'train', 'validation', 'test'
