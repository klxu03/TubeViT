from typing import Callable, Optional, Tuple, List

from torch import Tensor
from torchvision.datasets import UCF101

import torch
from torch.utils.data import Dataset
import os
import json
import math
import cv2
import numpy as np
import random

import re

class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label

class MyGodcaster(Dataset):
    def __init__(self, video_folder: str, caption_folder: str, tokenizer, vivit_processor, transform: Optional[Callable] = None) -> None:
        super().__init__()
        print("Initializing godcaster dataset")
        self.video_folder = video_folder
        self.caption_folder = caption_folder
        self.video_files = sorted(self.read_folder(self.video_folder, ".mp4"))
        self.caption_files = sorted(self.read_folder(self.caption_folder, "cleaned.json"))

        print("vid files and cap files done")

        index_lengths = []

        for caption_file in self.caption_files:
            with open(caption_file, "r") as f:
                data = json.load(f)
            index_lengths.append(len(data))

        print("caption files in")
        
        index_cumulative_lengths = list(np.cumsum(index_lengths))
        print(f"index cum lengths indices 0 to 5 {index_cumulative_lengths[:5]}")

        self.len = index_cumulative_lengths[-1]

        print("sums done")
    
        self.index_map = dict(zip(index_cumulative_lengths, list(zip(self.video_files, self.caption_files))))
        print(f"index maps indices 0 to 5, {list(self.index_map.items())[:5]}")

        print("index map done")

        self.tokenizer = tokenizer
        self.vivit_processor = vivit_processor
        self.transform = transform
        
    def read_folder(self, folder: str, ext: str) -> List[str]:
        total_files = [] 
        pattern = re.compile(r'round_\d+\.mp4$')  # This pattern matches 'round_' followed by digits and ending with '.mp4'

        for root, dirs, files in os.walk(folder):
            for name in files:
                if ext == ".mp4":
                    if pattern.match(name):
                        total_files.append(os.path.join(root, name))
                elif ext in name:
                    total_files.append(os.path.join(root, name))
        
        return total_files

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        og_index = index
        print("get_item og index", og_index)

        while index not in self.index_map:
            index -= 1

        print("index", index)

        file_pair = self.index_map[index]

        with open(file_pair[1]) as f:
            captions = json.load(f)
        print(f"captions, {captions[0]} and len {len(captions)}")

        container = cv2.VideoCapture(file_pair[0])
        FPS = container.get(cv2.CAP_PROP_FPS)
        # sentence_index = og_index - index
        sentence_index = random.randint(0, 10)
        print("sentence index", sentence_index)
        print("captions[sentence_index]['start']", captions[sentence_index]["start"])
        sentence_start_frame = math.ceil(captions[sentence_index]["start"] * FPS)
        print("sentence start frame", sentence_start_frame)

        if sentence_start_frame < 230:
            print(f"sentence start frame too small")
            num_copies = 230 // sentence_start_frame
            left_over = 230 % sentence_start_frame
            indicies = [0]*(num_copies + left_over)

            for i in range(1, sentence_start_frame):
                indicies += [i]*num_copies
        elif sentence_start_frame < FPS * 60:
            print(f"sentence start frame is less than a minute")
            indicies = list(np.linspace(0, sentence_start_frame, num=230))
        else:
            print(f"sentence start frame is big. we need to sample the frame indices")
            indicies = self.sample_frame_indices(sentence_start_frame, FPS)

        indicies = [sentence_start_frame-x for x in indicies][::-1]

        frames = []

        for index in indicies:
            container.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = container.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # input_tokens = self.tokenizer(" ".join([x["text"] for x in captions[:sentence_index]]), return_tensors="pt")
        # output_tokens = self.tokenizer(captions[sentence_index]["text"], return_tensors="pt")

        # Prepare frames into C T H W order for transformation
        frames = torch.stack([torch.from_numpy(f) for f in frames]) # T H W C right now
        print(f"original frames shape {frames.shape}")
        self.transform(frames)

        # return preprocessed_frames, input_tokens, output_tokens

        return frames, random.randint(1, 100)
    
    # Assumes we are over a minute long length
    def sample_frame_indices(self, frame_length_of_clip, FPS):
        initial_rate = FPS // 5
        ret = [i*initial_rate for i in range(50)]
        new_tot = frame_length_of_clip - math.ceil(10 * FPS)
        first_third = np.ceil(np.linspace(0, new_tot // 3, num=90) + math.ceil(10 * FPS))
        second_third = np.ceil(np.linspace(new_tot//3, new_tot, num=90) + math.ceil(10 * FPS))
        return ret + list(first_third) + list(second_third)