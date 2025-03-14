from kgcpy import *
import pandas as pd

joint_data = pd.read_csv('/test_data/Joined_test_metrics.csv')
centroid = joint_data['proj_centroid']
print(centroid.to_string())