import tables 
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from tools.mriutils import *
from tools.commoutils import load_pkl


class IXIDataset(Dataset):
    def __init__(self, data_dir, split='train', num_images=None, shuffle=False, corrupt_target=True, p_at_edge=0.025):
        super().__init__()
        self.corrupt_target = corrupt_target
        self.p_at_edge = p_at_edge

        if split == 'train':
            self.img, self.spec = load_pkl(os.path.join(data_dir, 'ixi_train.pkl'))
        else:
            self.img, self.spec = load_pkl(os.path.join(data_dir, 'ixi_valid.pkl'))
        
        if shuffle:
            perm = np.arange(self.img.shape[0])
            np.random.shuffle(perm)
            if num_images is not None:
                perm = perm[:num_images]
                self.img = self.img[perm]
                self.spec  = self.spec[perm]

        if num_images is not None:
            self.img = self.img[:num_images]
            self.spec  = self.spec [:num_images]

        print(self.img.shape, self.spec.shape)

    def __getitem__(self, idx):
        img = self.img[idx, :-1, :-1]
        assert img.dtype == np.uint8
        img = img.astype(np.float32) / 255.0 - 0.5
        spec = self.spec[idx, ...]

        spec_keep, keep_mask, source = sampleKspace(spec, self.p_at_edge)
        if self.corrupt_target:
            _, _, target = sampleKspace(spec, self.p_at_edge)
        else:
            target = img

        return {'orig_img': img, 'orig_spec': spec, 
                'spec_val': spec_keep, 'spec_mask': keep_mask,
                'source': source, 'target': target}

    def __len__(self):
        return len(self.img)


class BraTsDataset(Dataset):
    def __init__(self, data_path, split, 
                    modality=[0, 1, 2, 3], 
                    image_size=(256, 256), 
                    p_at_edge=0.1, 
                    transform=None):
        if split == 'train':
            filename = 'data_tr_233.h5'
        elif split == 'val':
            filename = 'data_val_50.h5'
        elif split == 'test':
            filename = 'data_test_51.h5'
        data_path = os.path.join(data_path, filename)  
        dataset = tables.open_file(data_path, "r") 
        self.truth = np.asarray(dataset.root.truth)
        self.truth = np.where(self.truth == 4, 3, self.truth)
        self.data = np.asarray(dataset.root.data)[:, modality, ...].astype(np.float32)
        self.subject_ids = dataset.root.subject_ids
        self.image_size = image_size
        self.p_at_edge = p_at_edge
        self.transform = transform

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx].decode("utf-8")
        image = self.data[idx, ...]
        truth = self.truth[idx, ...]
        
        # padding image to size
        padding_x = abs(image.shape[-1] - self.image_size[0]) // 2
        padding_y = abs(image.shape[-2] - self.image_size[1]) // 2
        
        image = np.pad(image, pad_width=((0, 0), (padding_x, padding_x), (padding_y, padding_y)), mode='constant', constant_values=0)
        truth = np.pad(truth, pad_width=((padding_x, padding_x), (padding_y, padding_y)), mode='constant', constant_values=0)
        image = (image.squeeze())[:-1, :-1]
        image = (image - 0.5) 
        spec = img2kspace(image)

        spec_keep, keep_mask, source = sampleKspace(spec, self.p_at_edge)
        _, _, target = sampleKspace(spec, self.p_at_edge)
        truth = truth.astype(np.uint8)[:-1, :-1]

        sample = {'orig_img':image, 'orig_spec':spec, 
                'source':source, 'target':target, 
                'spec_val': spec_keep, 'spec_mask': keep_mask,
                'truth':truth, 'subject_id':subject_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)


class MyDataset():
    def __init__(self,
                data_dir, 
                dataset ='ixi',
                modality=[2],
                p_at_Edge=0.9,
                batch_size=8, 
                num_workers=0,
                pin_memory=True):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if dataset == 'brats':
            self.train_dataset = BraTsDataset(data_dir, 'train', modality, p_at_edge=p_at_Edge)
            self.val_dataset = BraTsDataset(data_dir, 'val', modality, p_at_edge=p_at_Edge)
            self.test_dataset = BraTsDataset(data_dir, 'test', modality, p_at_edge=p_at_Edge)
        elif dataset =='ixi':
            self.train_dataset = IXIDataset(data_dir, 'train', shuffle=True, corrupt_target=True, p_at_edge=p_at_Edge)
            self.val_dataset = IXIDataset(data_dir, 'val', shuffle=False, corrupt_target=True, p_at_edge=p_at_Edge)

    def get_train_loader(self):
        return DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True,
                            pin_memory=self.pin_memory, drop_last=True)

    def get_val_loader(self):
        return DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=self.pin_memory, drop_last=True)

    def get_test_loader(self):
        return DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=self.pin_memory, drop_last=True) 

class ToTensor(object):
    def __call__(self, sample):
        return {'orig_image': torch.from_numpy(sample['orig_image']), 
                'orig_spec': torch.from_numpy(sample['orig_spec']), 
                'source': torch.from_numpy(sample['source']), 
                'target': torch.from_numpy(sample['target']), 
                'spec_keep': torch.from_numpy(sample['spec_keep']), 
                'keep_mask': torch.from_numpy(sample['keep_mask']),
                'truth': torch.from_numpy(sample['truth']), 
                'subject_id': sample['subject_id']}

if __name__ == '__main__':
    # dataset = MyDataset("D:/VAE/4modal2D_11slices", batch_size=1)
    # test_loader = dataset.get_test_loader()
    # for sample in test_loader:
    #     orig_image, orig_spec = sample['orig_image'], sample['orig_spec']
    #     source, target = sample['source'], sample['target']
    #     spec_keep, keep_mask = sample['spec_keep'], sample['keep_mask']
    #     print(source.min(), source.max())
    #     fig, axes = plt.subplots(3, 2)
    #     axes[0, 0].imshow(orig_image.squeeze().detach().cpu().numpy(), cmap='gray')
    #     axes[0, 1].imshow(np.log(np.abs(orig_spec.squeeze().detach().cpu().numpy())), cmap='gray')
    #     axes[1, 0].imshow(source.squeeze().detach().cpu().numpy(), cmap='gray')
    #     axes[1, 1].imshow(target.squeeze().detach().cpu().numpy(), cmap='gray')
    #     axes[2, 0].imshow(np.log(np.abs(spec_keep.squeeze().detach().cpu().numpy())), cmap='gray')
    #     axes[2, 1].imshow(keep_mask.squeeze().detach().cpu().numpy(), cmap='gray')
    #     plt.show()
    #     break
    dataset = MyDataset("./data/ixi-dataset", dataset='ixi', p_at_Edge=0.025, batch_size=16)
    train_loader = dataset.get_val_loader()
    for sample in train_loader:
        orig_image, orig_spec = sample['orig_img'], sample['orig_spec']
        source, target = sample['source'], sample['target']
        spec_val, spec_mask = sample['spec_val'], sample['spec_mask']

        # print(psnr(source.squeeze().detach().cpu().numpy(), orig_image.squeeze().detach().cpu().numpy()))
        # print(psnr(target.squeeze().detach().cpu().numpy(), orig_image.squeeze().detach().cpu().numpy()))

        print(source.size(), target.size())

        # fig, axes = plt.subplots(3, 2)

        # axes[0, 0].imshow(orig_image.squeeze().detach().cpu().numpy(), cmap='gray')
        # axes[0, 1].imshow(np.log(np.abs(orig_spec.squeeze().detach().cpu().numpy())), cmap='gray')
        # axes[1, 0].imshow(source.squeeze().detach().cpu().numpy(), cmap='gray')
        # axes[1, 1].imshow(target.squeeze().detach().cpu().numpy(), cmap='gray')
        # axes[2, 0].imshow(np.log(np.abs(spec_val.squeeze().detach().cpu().numpy())), cmap='gray')
        # axes[2, 1].imshow(spec_mask.squeeze().detach().cpu().numpy(), cmap='gray')
        # plt.show()
        # break