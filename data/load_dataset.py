import argparse
from pathlib import Path

class ScannetDataset:
    def __init__(self, file_path, num_scene_batch=5, num_shffule_per_iter=10): 

        self.num_scene_batch = num_scene_batch
        self.num_shffule_per_iter = num_shffule_per_iter
        # [(scene_name, [ordered_frame_names]), (), (), ...]
        self.ordered_batches = []
        self.sub_batches = []
        with open(file_path, 'r') as f:
            self.used_subscenes = f.readlines()
            self.scene_names = []
            self.frame_names = {}
            curr_scene_index = 0
            curr_scene_name = self.used_subscenes[0].strip().split('/')[0]
            curr_batch_name = f"batch_{curr_scene_index}"
            # index = 0
            for i in range(len(self.used_subscenes)):
                full_path = self.used_subscenes[i].strip()
                scene_name = '/'.join(full_path.split('/')[:2])
                frame_name = full_path.split('/')[-1]
                if scene_name not in self.scene_names:
                    # self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
                    self.frame_names[scene_name] = {} 
                    self.scene_names.append(scene_name) 
                    curr_scene_index = 0 
                if (curr_scene_index % self.num_scene_batch == 0) or (curr_scene_index == 0): 
                    curr_batch_name = f"batch_{curr_scene_index}" 
                    self.frame_names[scene_name][curr_batch_name] = []
                    if len(self.sub_batches) > 0:
                        if len(self.sub_batches) < self.num_scene_batch:
                            self.patch_batch()
                        self.ordered_batches.append((curr_scene_name, self.sub_batches.copy()))  # .copy() is must to avoid shallow copy
                        self.sub_batches.clear()
                        # print(self.ordered_batches[index], index, len(self.ordered_batches))
                        # index += 1
                    if curr_scene_index == 0: 
                        curr_scene_name = scene_name  # use 'curr_scene_name' to collect last frames in last scene
                self.frame_names[scene_name][curr_batch_name].append(frame_name)
                self.sub_batches.append(frame_name)
                curr_scene_index += 1

        import random
        random.shuffle(self.ordered_batches)
        self.shuffle_batches = [
            self.ordered_batches[i:i+self.num_shffule_per_iter]
            for i in range(0, len(self.ordered_batches), num_shffule_per_iter)
        ]

        print(len(self.shuffle_batches))
        for i, group in enumerate(self.shuffle_batches):
            print(f"Iteration group {i}:")
            for batch in group:
                scene_name = batch[0].split('/')[1]
                print(f"scene_name: {scene_name}, frames: {batch[1]}")
    
    def patch_batch(self):
        latest_batch = self.ordered_batches[-1][1]
        curr_batch = self.sub_batches.copy()
        delta_num = self.num_scene_batch - len(curr_batch)
        self.sub_batches = latest_batch[-delta_num:] + self.sub_batches

 
def main(args):
    root = Path(args.root)
    file_path = root / Path(args.file)
    custom_dataloader = ScannetDataset(file_path, num_scene_batch=10, num_shffule_per_iter=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='lidar Visualization')
    parser.add_argument('root', type=str, default='/EmbodiedOcc/data/occscannet')
    parser.add_argument('file', type=str, default='train_mini_final.txt')
    args = parser.parse_args()

    main(args)



