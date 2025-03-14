import pandas as pd
import re

test_csv_loc = ('/data/test/metadata.csv')
test_ext_csv_loc = ('/data/test/metadata_extended.csv')

predicted_test_loc = ('/test_data/artifacts/AMG_adaptiveDropout_test_metrics.csv')
test_metrics = pd.read_csv(predicted_test_loc)
test_metrics = test_metrics.drop(columns=['Unnamed: 0'])

test_metrics['Accuracy'] = test_metrics['Accuracy'].str.replace(r'[^A-Za-z0-9.]', '', regex=True)
test_metrics['F1Score'] = test_metrics['F1Score'].str.replace(r'[^A-Za-z0-9.]', '', regex=True)
test_metrics['Precision'] = test_metrics['Precision'].str.replace(r'[^A-Za-z0-9.]', '', regex=True)
test_metrics['Recall'] = test_metrics['Recall'].str.replace(r'[^A-Za-z0-9.]', '', regex=True)
test_metrics['MeanIOU'] = test_metrics['MeanIOU'].str.replace(r'[^A-Za-z0-9.]', '', regex=True)

test_metrics['Accuracy'] = test_metrics['Accuracy'].str.replace(r'tensor', '', regex=True)
test_metrics['F1Score'] = test_metrics['F1Score'].str.replace(r'tensor', '', regex=True)
test_metrics['Precision'] = test_metrics['Precision'].str.replace(r'tensor', '', regex=True)
test_metrics['Recall'] = test_metrics['Recall'].str.replace(r'tensor', '', regex=True)
test_metrics['MeanIOU'] = test_metrics['MeanIOU'].str.replace(r'tensor', '', regex=True)


#print(test_metrics.to_string())

test_csv = pd.read_csv(test_csv_loc)
test_csv_extended = pd.read_csv(test_ext_csv_loc)

## Join Dataframes
joined_test = test_csv.join(test_metrics)
print(joined_test.to_string())


joined_test.to_csv("/test_data/Joined_test_metrics.csv", index=False)
assert(False)
#print(joined_test.to_string())

cloudFree_Diff1 = len(test_csv[(test_csv['cloud_coverage'] == 'cloud-free') & (test_csv['difficulty'] == 1.0)])
cloudFree_Diff2 = len(test_csv[(test_csv['cloud_coverage'] == 'cloud-free') & (test_csv['difficulty'] == 2.0)])
cloudFree_Diff3 = len(test_csv[(test_csv['cloud_coverage'] == 'cloud-free') & (test_csv['difficulty'] == 3.0)])
cloudFree_Diff4 = len(test_csv[(test_csv['cloud_coverage'] == 'cloud-free') & (test_csv['difficulty'] == 4.0)])
# print("cloudFree_Diff1: ", cloudFree_Diff1)
# print("cloudFree_Diff2: ", cloudFree_Diff2)
# print("cloudFree_Diff3: ", cloudFree_Diff3)
# print("cloudFree_Diff4: ", cloudFree_Diff4)

almostClear_Diff1 = len(test_csv[(test_csv['cloud_coverage'] == 'almost-clear') & (test_csv['difficulty'] == 1.0)])
almostClear_Diff2 = len(test_csv[(test_csv['cloud_coverage'] == 'almost-clear') & (test_csv['difficulty'] == 2.0)])
almostClear_Diff3 = len(test_csv[(test_csv['cloud_coverage'] == 'almost-clear') & (test_csv['difficulty'] == 3.0)])
almostClear_Diff4 = len(test_csv[(test_csv['cloud_coverage'] == 'almost-clear') & (test_csv['difficulty'] == 4.0)])
# print("almostClear_Diff1: ", almostClear_Diff1)
# print("almostClear_Diff2: ", almostClear_Diff2)
# print("almostClear_Diff3: ", almostClear_Diff3)
# print("almostClear_Diff4: ", almostClear_Diff4)

lowCloudy_Diff1 = len(test_csv[(test_csv['cloud_coverage'] == 'low-cloudy') & (test_csv['difficulty'] == 1.0)])
lowCloudy_Diff2 = len(test_csv[(test_csv['cloud_coverage'] == 'low-cloudy') & (test_csv['difficulty'] == 2.0)])
lowCloudy_Diff3 = len(test_csv[(test_csv['cloud_coverage'] == 'low-cloudy') & (test_csv['difficulty'] == 3.0)])
lowCloudy_Diff4 = len(test_csv[(test_csv['cloud_coverage'] == 'low-cloudy') & (test_csv['difficulty'] == 4.0)])
# print("lowCloudy_Diff1: ", lowCloudy_Diff1) ## No values for low cloudy and difficulty 1.0
# print("lowCloudy_Diff2: ", lowCloudy_Diff2)
# print("lowCloudy_Diff3: ", lowCloudy_Diff3)
# print("lowCloudy_Diff4: ", lowCloudy_Diff4)

midCloudy_Diff1 = len(test_csv[(test_csv['cloud_coverage'] == 'mid-cloudy') & (test_csv['difficulty'] == 1.0)])
midCloudy_Diff2 = len(test_csv[(test_csv['cloud_coverage'] == 'mid-cloudy') & (test_csv['difficulty'] == 2.0)])
midCloudy_Diff3 = len(test_csv[(test_csv['cloud_coverage'] == 'mid-cloudy') & (test_csv['difficulty'] == 3.0)])
midCloudy_Diff4 = len(test_csv[(test_csv['cloud_coverage'] == 'mid-cloudy') & (test_csv['difficulty'] == 4.0)])
# print("midCloudy_Diff1: ", midCloudy_Diff1) ## No values for mid cloudy and difficulty 1.0
# print("midCloudy_Diff2: ", midCloudy_Diff2)
# print("midCloudy_Diff3: ", midCloudy_Diff3)
# print("midCloudy_Diff4: ", midCloudy_Diff4)

cloudy_Diff1 = len(test_csv[(test_csv['cloud_coverage'] == 'cloudy') & (test_csv['difficulty'] == 1.0)])
cloudy_Diff2 = len(test_csv[(test_csv['cloud_coverage'] == 'cloudy') & (test_csv['difficulty'] == 2.0)])
cloudy_Diff3 = len(test_csv[(test_csv['cloud_coverage'] == 'cloudy') & (test_csv['difficulty'] == 3.0)])
cloudy_Diff4 = len(test_csv[(test_csv['cloud_coverage'] == 'cloudy') & (test_csv['difficulty'] == 4.0)])
# print("cloudy_Diff1: ", cloudy_Diff1) 
# print("cloudy_Diff2: ", cloudy_Diff2)
# print("cloudy_Diff3: ", cloudy_Diff3)
# print("cloudy_Diff4: ", cloudy_Diff4)




#print(test_csv_extended.to_string())
#test_csv.to_csv('/xarray/test/metadata.csv', index=False)