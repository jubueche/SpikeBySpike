import numpy as np  
import os
from scipy.io.wavfile import read
from collections import defaultdict
import matplotlib.pyplot as plt
from free_spoken_digit_dataset.utils import fsdd, spectogramer
from scipy.signal import resample


class AudioHelper:

    def __init__(self):

        self.recordings_path = os.path.join(os.getcwd(),  "free_spoken_digit_dataset/recordings/")
        self.spectogram_path = os.path.join(os.getcwd(),  "free_spoken_digit_dataset/spectograms/")
        self.data = fsdd.FSDD(self.recordings_path)
        self.training_idx = 0
        self.testing_idx = 0
        self.data_test = defaultdict(list)
        self.data_train = defaultdict(list)
        self.split_already = False
        self.train_number = 0
        self.test_number = 0

    def fill_spectograms(self):
        spectogramer.dir_to_spectrogram(recordings_path, spectogram_path)
        specs, labels = fsdd.FSDD.get_spectrograms(spectogram_path)

    def split_train_test(self, train_ratio=0.8):

        digit_nums = np.zeros(10)
        for digit in range(0,10):
            digit_nums[digit] = len(self.data.recording_paths[digit])
        
        for idx,num in enumerate(digit_nums):
            # Create a linspace and shuffle
            indices = np.linspace(0,digit_nums[idx]-1,digit_nums[idx]).astype(np.int)
            np.random.shuffle(indices)
            train_indices = indices[0:int(0.8*digit_nums[idx])].astype(np.int)
            test_indices = indices[int(0.8*digit_nums[idx])+1:].astype(np.int)
            self.train_number += int(0.8*digit_nums[idx])
            self.test_number += int(0.2*digit_nums[idx])
            self.data_train[idx] = [self.data.recording_paths[idx][i] for i in train_indices]
            self.data_test[idx] = [self.data.recording_paths[idx][i] for i in test_indices]
        self.split_already = True

    def get_random_sample(self):
        if(not self.split_already):
            self.split_train_test()
        digit = np.random.randint(0,10)
        t,val = read(os.path.join(os.getcwd(), self.data_train[digit][self.training_idx]))
        return t,val

    def get_next_training(self, digit=-1, length = 500):
        if(not self.split_already):
            self.split_train_test()

        if(digit==-1):
            digit = np.random.randint(0,10)
        
        sig = np.zeros(length)
        too_long = True
        while(too_long):
            t,val = read(os.path.join(os.getcwd(), self.data_train[digit][self.training_idx]))
            self.training_idx += 1
            self.training_idx = self.training_idx % len(self.data_train[digit])
            factor = int(t/1000)
            subsamples_n = int(len(val) / factor)
            # val = val[::subsample]
            val = resample(val, subsamples_n)
            if(len(val) < length):
                sig[0:len(val)] = val
                too_long = False

        sig /= (max(sig)-min(sig))

        return digit, sig

    def get_next_test(self, digit=-1, length = 500):
        if(not self.split_already):
            self.split_train_test()
            
        if(digit==-1):
            digit = np.random.randint(0,10)

        sig = np.zeros(length)
        too_long = True
        while(too_long):
            t,val = read(os.path.join(os.getcwd(), self.data_test[digit][self.testing_idx]))
            self.testing_idx += 1
            self.testing_idx = self.testing_idx % len(self.data_test[digit])
            factor = int(t/1000)
            subsamples_n = int(len(val) / factor)
            #val = val[::subsample]
            val = resample(val, subsamples_n)
            if(len(val) < length):
                sig[0:len(val)] = val
                too_long = False

        sig /= (max(sig)-min(sig))

        return digit, sig