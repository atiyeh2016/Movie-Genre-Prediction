from FilmDescriptionsDataset import FilmDescriptionsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ToTensor import ToTensor

sentence_length = 250
transformed_dataset = FilmDescriptionsDataset('dataset.cleaned.csv', sentence_length,
                                              transform=transforms.Compose([ToTensor()]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=False, num_workers=0)
c = 0
for i_batch, sample_batched in enumerate(dataloader):
    c += 1
    print(sample_batched['description'].size())
    print(sample_batched['genre'].size())
    if c > 5: break