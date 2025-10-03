import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import default_collate

## ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def trainTransforms(
    imgSize=(224, 224),
    img_mean=IMAGENET_MEAN,
    img_std=IMAGENET_STD,
    interpolation=InterpolationMode.BILINEAR,
    hflip_prob=0.5,
    random_aug_magnitude=9,
):
    transformation_chain = []
    transformation_chain.append(
        v2.RandomResizedCrop(imgSize, interpolation=interpolation, antialias=True)
    )  # grab different chunks of the image
    if hflip_prob > 0:
        transformation_chain.append(
            v2.RandomHorizontalFlip(hflip_prob)
        )  # flip the image horizontally with a probability of 0.5
    if random_aug_magnitude > 0:
        transformation_chain.append(
            v2.RandAugment(magnitude=random_aug_magnitude, interpolation=interpolation)
        )  # randomly augment the image
    transformation_chain.append(v2.PILToTensor())  # convert the image to a tensor
    transformation_chain.append(
        v2.ToDtype(torch.float32, scale=True)
    )  # convert the image to float type
    transformation_chain.append(
        v2.Normalize(mean=img_mean, std=img_std)
    )  # normalize the image with the given mean and std
    return transforms.Compose(transformation_chain)


print(trainTransforms())


def evalTransforms(
    imgSize=(224, 224),
    ResizeSize=(256, 256),
    img_mean=IMAGENET_MEAN,
    img_std=IMAGENET_STD,
    interpolation=InterpolationMode.BILINEAR,
):
    transformation = transforms.Compose(
        [
            v2.Resize(
                ResizeSize, interpolation=interpolation, antialias=True
            ),  # resize the image to 256x256
            v2.CenterCrop(imgSize),  # crop the center of the image to 224x224
            v2.PILToTensor(),  # convert the image to a tensor
            v2.ToDtype(torch.float32, scale=True),  # convert the image to float type
            v2.Normalize(
                mean=img_mean, std=img_std
            ),  # normalize the image with the given mean and std
        ]
    )
    return transformation


# We need to do cutmix and mixup in a custom collate function
# we do this between consecutive images, example is take a piece of image 1 and a piece of image 2 and combine them to make a new image or take a piece of image 1 and paste it on image 2 to make a new image
def Mix_cut_collate_fn(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000):
    mix_cut_transforms = None
    mix_cut = []
    if mixup_alpha > 0:
        mix_cut.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mix_cut.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if len(mix_cut) > 0:
        mix_cut_transforms = v2.RandomChoice(mix_cut)

    def collate_fn(batch):
        collated = default_collate(batch)
        if mix_cut_transforms is not None:
            collated = mix_cut_transforms(collated)
        return collated

    return collate_fn


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]
        if target.ndim == 2:
            target = target.argmax(dim=-1)
        values, pred = output.topk(
            maxk, dim=-1, largest=True, sorted=True
        )  # pred shape is (batch_size, maxk)
        pred = pred.transpose(0, 1)
        correct = pred == target
        accs = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            accs.append((correct_k) / batch_size)
        if len(accs) == 1:
            return accs[0]
        else:
            return accs
