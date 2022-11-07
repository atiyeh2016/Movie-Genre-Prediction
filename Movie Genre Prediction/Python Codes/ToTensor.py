import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'description': torch.from_numpy(sample['description']).type(torch.FloatTensor),
                'genre': sample['genre']}