from pathlib import Path
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import os
import PIL
import numpy as np

ROOT = '/media/mountHDD3/data_storage'

transform = A.Compose([
            A.CenterCrop(height=148, width=148),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, angle_in_degrees=True))

ex_transform = A.Compose([
               A.CenterCrop(height=148, width=148),
               A.ToFloat(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, angle_in_degrees=True))


class Celeb_A(CelebA):
    def __init__(self, root: str | Path, split: str = "train", target_type: torch.List[str] | str = "attr", transform: None = None, 
                target_transform: None = None, download: bool = False) -> None:
         super().__init__(root, split, target_type, transform, target_transform, download)

    def __getitem__(self, index):
        image = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        landmarks = self.landmarks_align[index, :].to(dtype=torch.float32)
        landmarks=landmarks.reshape(5,2).tolist()
    
        if self.transform is not None:
            transformed = self.transform(image=np.array(image), keypoints=landmarks)
            transformed_image = transformed['image']
            transformed_landmarks = transformed['keypoints']
            image = torch.from_numpy(transformed_image).permute(2, 0, 1).float()
            img_height, img_width = transformed_image.shape[:2]
            normalized_landmarks = transformed_landmarks / np.array([img_width, img_height])
            landmarks = torch.tensor(normalized_landmarks, dtype=torch.float32).flatten()

        return image, landmarks


def get_celeba(args):
        train_ds = Celeb_A(root=ROOT, split='train', transform=transform, target_type='landmarks')
        valid_ds = Celeb_A(root=ROOT, split='valid', transform=ex_transform, target_type='landmarks')
        test_ds = Celeb_A(root=ROOT, split='test', transform=ex_transform, target_type='landmarks')

        # def generate_batch(data_batch):
        #     image_batch=[]
        #     landmarks_batch=[]
        #     for(image, landmarks) in data_batch:
        #         # image = transform(image)
        #         image_batch.append()
        #         landmarks = torch.tensor(landmarks, dtype=torch.float32)
        #         landmarks_batch.append(landmarks)
        #     image_batch = torch.stack(image_batch)
        #     landmarks_batch = torch.stack(landmarks_batch)
        #     return image_batch, landmarks_batch

        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
        valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
        test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)

        args.num_landmarks=10

        args.num_train_sample = len(train_ds)
        args.num_valid_sample = len(valid_ds)
        args.num_test_sample = len(test_ds)
        args.num_train_batch = len(train_dl)
        args.num_valid_batch = len(valid_dl)
        args.num_test_batch = len(test_dl)

        return (train_dl, valid_dl, test_dl, args)

# if __name__ == '__main__':
#      dataset = Celeb_A(root=ROOT, split='valid', transform=ex_transform, target_type='landmarks')
#      image, landmarks = dataset[11]
#      print (landmarks)

     