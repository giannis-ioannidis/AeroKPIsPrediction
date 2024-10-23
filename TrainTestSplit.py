import re
import os

test_path = 'home/gradguest/gradguest_data/Tests/FDR_V6_tests'

def test_list(test_path):

  test_files = os.listdir(test_path)
  lst = []

  for file_name in test_files:
    temp = re.findall(r'SIMBAD_(\d+)', file_name)
    lst.append(temp)

  return lst

test_traj = test_list(test_path)

dir_path = 'home/gradguest/gradguest_data/PprVariables/FDR_weather_phases_dsc'

def train_test_split(dir_path):

  for test_name in test_traj:

    for file_name in directory_files:

      temp = re.findall(r'SIMBAD_(\d+)', file_name)

      if temp = test_name:

        os.replace(dir_path + '/' + file_name, 'home/gradguest/gradguest_data/Tests/TestSet')

      else:

        os.replace(dir_path + '/' + file_name, 'home/gradguest/gradguest_data/Tests/TrainSet')
