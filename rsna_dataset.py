import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms
from timm.data import create_transform
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:
    img_size = (1024, 1024)
    flip_breast = False
    target = "cancer"

#https://www.kaggle.com/code/paulbacher/custom-preprocessor-rsna-breast-cancer/notebook

# Determine the current breast side
def _determine_breast_side(img):
    col_sums_split = np.array_split(np.sum(img, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])
    if left_col_sum > right_col_sum:
        return 'L'
    else:
        return 'R'  

# Flip the breast horizontally on the chosen side 
def _flip_breast_side(img):
    breast_side = 'L' #default
    img_breast_side = _determine_breast_side(img)
    if img_breast_side == breast_side:
        return img
    else:
        return np.fliplr(img)   
'''
def get_transform(stage):
    rand_num = 0.9 #np.random.rand()
    if stage:
        if rand_num <= 0.3:
            transform = create_transform(
                input_size=(CFG.img_size[0], CFG.img_size[1]),
                is_training=True,
                auto_augment='augmix-m5-w4-d2',
                interpolation="random",
            )
        elif rand_num > 0.3 and rand_num <=0.5:
            transform = create_transform(
                input_size=(CFG.img_size[0], CFG.img_size[1]),
                is_training=True,
                auto_augment='rand-mstd1-w0',
                interpolation="random",
            )
            
        elif rand_num > 0.5 and rand_num <0.8:
            transform = create_transform(
                input_size=(CFG.img_size[0], CFG.img_size[1]),
                is_training=True,
                auto_augment='rand-m9-mstd0.5',
                interpolation="random",
            )
        else:
            transform = create_transform(
                input_size=(CFG.img_size[0], CFG.img_size[1]),
                is_training=True,
                interpolation="bilinear",
            )
    else:
        transform = create_transform(
            input_size=(CFG.img_size[0], CFG.img_size[1]),
            is_training=False,
            interpolation="bilinear",
        )
    return transform


def get_transform(stage):
    if stage == "train":
        rand_num = np.random.rand()
        if rand_num <= 0.5:
            vision_Transformation = torch_transforms.Compose([
                torch_transforms.ToPILImage(),
#                 torch_transforms.CenterCrop(CFG.img_size[0]/2),
                torch_transforms.Resize(size=(CFG.img_size[0], CFG.img_size[1])),
                
                torch_transforms.RandomHorizontalFlip(p=0.5),
                torch_transforms.RandomRotation(degrees=(-5, 5)), 
    #             torch_transforms.RandAugment(num_ops = 2, magnitude = 10, num_magnitude_bins = 31),
                torch_transforms.ToTensor(),
#                 torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            vision_Transformation = torch_transforms.Compose([
                torch_transforms.ToPILImage(),
#                 torch_transforms.CenterCrop(CFG.img_size[0]/2),
                torch_transforms.Resize(size=(CFG.img_size[0], CFG.img_size[1])),
#                 torch_transforms.RandomInvert(p=0.5),
                torch_transforms.RandomVerticalFlip(p=0.5),
                
                torch_transforms.RandomRotation(degrees=(-15, 15)),
                
#                 torch_transforms.TenCrop(CFG.img_size[0]),
                
#                 torch_transforms.Lambda(lambda crops: [torch_transforms.ToTensor()(crop) for crop in crops]),
                
                torch_transforms.ColorJitter(
                    brightness = 0.175,   
                    contrast = 0.175,   
                    saturation = 0.195,   
                    hue = (0.1, 0.25)), 

#                 torch_transforms.RandAugment(num_ops = 2, magnitude = 10, num_magnitude_bins = 31),
                torch_transforms.ToTensor(),
                
#                 torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return vision_Transformation
    else:
        vision_Transformation = torch_transforms.Compose([
            torch_transforms.ToPILImage(),
#             torch_transforms.CenterCrop(CFG.img_size[0]/2),
            torch_transforms.Resize(size=(CFG.img_size[0], CFG.img_size[1])),
                
            torch_transforms.ToTensor(),
#             torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return vision_Transformation
'''
    
def get_transform(stage):
    if stage == 'train':
        return A.Compose([
#                 A.AdvancedBlur(always_apply=False, p=1.0, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),                  rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
            A.HorizontalFlip(),
#             A.VerticalFlip(),
            A.CenterCrop(always_apply=False, p=1.0, height=CFG.img_size[0], width=CFG.img_size[1]),
#             A.augmentations.geometric.resize.Resize(CFG.img_size[0],CFG.img_size[1]),
#             A.geometric.transforms.Affine(scale=(0.2, 1.0), translate_percent=(0.1, 0.1), translate_px=None, rotate=(2,2),                              shear=10, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0,                                                   fit_output=False,keep_ratio=False, always_apply=False, p=0.5),
            

            A.dropout.coarse_dropout.CoarseDropout (max_holes=10, max_height=48, max_width=48, min_holes=None,                              min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
#             A.Flip(p=0.5),
            
#             A.geometric.rotate.SafeRotate(limit=5, interpolation=1, border_mode=4, p=0.5),
#                 A.RandomBrightnessContrast(p=0.1),

             A.ColorJitter (brightness=0.35, contrast=0.35, saturation=0.38, hue=0.32, always_apply=False, p=0.4),

             A.Normalize(
                mean=0,
                std=1,
            ),
            ToTensorV2(),
        ])
    
    elif stage == 'valid':
        return A.Compose([
            A.CenterCrop(always_apply=False, p=1.0, height=CFG.img_size[0], width=CFG.img_size[1]),
#             A.augmentations.geometric.resize.Resize(CFG.img_size[0],CFG.img_size[1]),

            A.Normalize(
                mean=0,
                std=1,
            ),
            ToTensorV2(),
        ])
    
def img2roi(img, is_dicom=False):
    """
    Returns ROI area in other words 
    cuts the image to a desired one
    
    Because there are machine label tags,
    undesired details out of the breast image.
    """
    if not is_dicom:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = np.array(img * 255, dtype = np.uint8)
    # Binarize the image
    bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Find ROI from largest contour
    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    roi =  img[np.min(xs):np.max(xs), np.min(ys):np.max(ys)]
#     roi = cv2.resize(roi, (1024,1024), cv2.INTER_LINEAR)

#     print(f"Shape of ROI image: {roi.shape}")
    
    return roi



class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.paths = df['path'].values
        self.transforms = transforms
        self.targets = df['cancer'].values

    def __len__(self):
        return len(self.paths)
    
    def get_labels(self):   return self.targets
    
    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """ 
            
        try:
#             image = cv2.imread(self.paths[idx],cv2.IMREAD_UNCHANGED)
            image = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE) # 1 channel
            shape = image.shape
            if len(image.shape) == 2:
                image = image[:,:,None]
#             image = cv2.cvtColor(cv2.imread(self.paths[idx],cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
#             image = np.asarray(Image.open(self.paths[idx]).convert('RGB'))

            if(CFG.flip_breast):
                try:
                    image = _flip_breast_side(image)
                except Exception as ex:
                    print("failed flipping ", ex)

    
#             image = (image * 255).astype(np.uint8)
#             image = img2roi(image, is_dicom=False)
#             image = cv2.resize(image,(CFG.img_size[0],CFG.img_size[1]), interpolation=cv2.INTER_LINEAR)
 
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        

        except Exception as ex:
            print(self.paths[idx], ex)
            return None
        
#         image = image.astype(np.float32)
   
        if self.transforms=='valid':
            valid_aug = get_transform(stage='valid')
#             image = valid_aug(image)
            image = valid_aug(image=image)['image']
        if self.transforms=='train':
            train_aug = get_transform(stage='train')
#             image = train_aug(image)
            image = train_aug(image=image)['image']
       
        image = image / 255
        
        if CFG.target in self.df.columns:
            target = torch.as_tensor(self.df.iloc[idx].cancer)
            return image, target
      
        return image
'''       



class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.paths = df['path'].values
        self.transforms = transforms
        self.targets = df['cancer'].values

    def __len__(self):
        return len(self.paths)
    
    def get_labels(self):   return self.targets
    
    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """ 
        try:
            #image = cv2.imread(self.paths[idx])
            image = Image.open(self.paths[idx]).convert('RGB')
        except Exception as ex:
            print(self.paths[idx], ex)
            return None
        #image = img2roi(image, is_dicom=False)
        if self.transforms=='valid':
            valid_aug = get_transform(stage='valid')
            image = torch.as_tensor(valid_aug(image),dtype=torch.float32)
            #image = valid_aug(image)
        if self.transforms=='train':
            train_aug = get_transform(stage='train')
            image = torch.as_tensor(train_aug(image),dtype=torch.float32)
            #image = train_aug(image)
         
        
        
        if CFG.target in self.df.columns:
            target = torch.as_tensor(self.df.iloc[idx].cancer)
            return image, target
      
        return image
 '''