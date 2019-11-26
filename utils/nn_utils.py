from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2 
import random

class ContrastiveLossD(torch.nn.Module):
    """ Contrastive loss function for Siamese nets.
    
        The forward path step takes 2 embeddings of different images and label. 
        Calculates a loss function based on the Euclidean distance from embeddings of one image from another.
        Label = 1 -> labels of classes of initial images identical and distance should be minimum
        Based on: https://github.com/delijati/pytorch-siamese/
        
        Attributes:
            margin: threshold, if the images from different classes 
            and the distance is greater than the margin, we do not penalize 
            if dist>=margin and label = 0 -> loss =0
        """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class ContrastiveLossLecun(torch.nn.Module):
    """ Contrastive loss function for Siamese nets.
    
        The forward path step takes 2 embeddings of different images and label. 
        Calculates a loss function based on the Euclidean distance from embeddings of one image from another.
        Label = 0 -> labels of classes of initial images identical and distance should be minimum
        label = 1 -> different classes, maximize distance 
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        
        Attributes:
            margin: threshold, if the images from different classes 
            and the distance is greater than the margin, we do not penalize 
            if dist>=margin and label = 1 -> loss =0
   
        """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive




def get_image(im_path, transform=None):
    
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    # image is a np.array  with shape (height,width, channels)
    if transform:
        augmented = transform(image=image)
        image = augmented["image"]
    return image


def make_oneshot_task(N, dataloader):
    """This function creates a set of pairs from the test image (anchor) and the auxiliary image
    one of the auxiliary images has the same class (label=1), the others are opposite classes.(label=0)

    Args:
        N- int, number of pairs
        dataloader - pytorch dataloader based on SiameseNetworkDataset

    Return: 
        Torch.Tensor with acnhor images
        torch.Tensor with support images
        torch.Tensor with labels
    """
    # os.path.join(root_dir, image_name)

    cat_to_im_paths = dataloader.dataset.cat_to_im_paths

    # random choice anchor class
    anchor_class = np.random.choice(list(cat_to_im_paths.keys()))

    # random choice anchor image from ancor class
    list_of_same_images = list(cat_to_im_paths[anchor_class])
    anchor_im_path = np.random.choice(list_of_same_images)
    anch_im = get_image(anchor_im_path, dataloader.dataset.transform)

    # create batch of N anchor images
    # batch_anch_ims = [deepcopy(a) for x in range(N)]
    batch_anch_ims = [anch_im for x in range(N)]

    # dowload image with similar class to anchor
    similar_image = [x for x in list_of_same_images if x != anchor_im_path]
    similar_im_path = np.random.choice(similar_image)
    similar_im = get_image(similar_im_path, dataloader.dataset.transform)

    # create batch of "seconds" images, one from same class and N-1 from opposite
    batch_second_ims = [similar_im]
    batch_labels = [1.0]
    # download n-1 images from opposite class
    list_of_opposite_keys = [x for x in cat_to_im_paths.keys() if x != anchor_class]

    for i in range(N - 1):
        batch_labels.append(0.0)
        # select random opposite class
        opp_class = np.random.choice(list_of_opposite_keys)
        # choice random image from selected class
        opp_im_path = np.random.choice(list(cat_to_im_paths[opp_class]))
        opp_im = get_image(opp_im_path, dataloader.dataset.transform)
        batch_second_ims.append(opp_im)

    batch_anch_ims, batch_second_ims, batch_labels = shuffle(
        batch_anch_ims, batch_second_ims, batch_labels
    )
    return (
        torch.stack(batch_anch_ims),
        torch.stack(batch_second_ims),
        torch.Tensor(batch_labels),
    )




def test_oneshot_L2(model, N, k, dataloader, device, verbose=0):
    """Test average N way oneshot learning accuracy of
    a siamese neural net over k one-shot tasks.
    Based on euclidian distance (L2).
    
    Args:
        model: neural net, generated 2 output embeddings
        N: int, number of samples in each one-shot task
        k: int, number of one shot tasks
        dataloader: pytorch dataloader
        device: str, cuda of cpu
        verbose: bool, print percent correct or not

    Return: percent of correct one short tasks
    """
    n_correct = 0
    if verbose:
        print(
            "Evaluating model on {} random {} way one-shot learning tasks ... \n".format(
                k, N
            )
        )
    for i in range(k):
        img0, img1, targets = make_oneshot_task(N, dataloader)
        img0, img1, targets = img0.to(device), img1.to(device), targets.to(device)
        emb_anchor, emb_images = model(img0, img1)
        eucledian_distance = F.pairwise_distance(emb_anchor, emb_images)
        if torch.argmin(eucledian_distance) == torch.argmax(targets):
            n_correct += 1

    percent_correct = 100.0 * n_correct / k
    if verbose:
        print(
            "Got an average of {:.1f}% {} way one-shot learning accuracy \n".format(
                percent_correct, N
            )
        )
    return percent_correct


def test_oneshot_prob_single_neuron(model, N, k, dataloader,device, verbose=0):
    """Test average N way oneshot learning accuracy of
    a siamese neural net over k one-shot tasks.
    Based on probability of class 1.
    
    Args:
        model: neural net, ouput - probability of class 1
        N: int, number of samples in each one-shot task
        k: int, number of one shot tasks
        dataloader: pytorch dataloader
        device: str, cuda of cpu
        verbose: bool, print percent correct or not

    Return: percent of correct one short tasks
    """
    n_correct = 0
    if verbose:
        print(
            "Evaluating model on {} random {} way one-shot learning tasks ... \n".format(
                k, N
            )
        )
    for i in range(k):
        img0, img1, targets = make_oneshot_task(N, dataloader)
        img0, img1, targets = img0.to(device), img1.to(device), targets.to(device)

        output = model(img0, img1)
        prob_of_same = output

        if torch.argmax(prob_of_same) == torch.argmax(targets):
            n_correct += 1

    percent_correct = 100.0 * n_correct / k
    if verbose:
        print(
            "Got an average of {:.1f}% {} way one-shot learning accuracy \n".format(
                percent_correct, N
            )
        )
    return percent_correct


def test_oneshot_cross_entropy(model, N, k, dataloader,device, verbose=0):
    """Test average N way oneshot learning accuracy of
    a siamese neural net over k one-shot tasks.
    Based on the fact that in the column with index 1
    lies the logit of class 1 (images from the same class)
    
    Args:
        model: neural net, generated 2 raw logits, first for class 0, second for class 1
        N: int, number of samples in each one-shot task
        k: int, number of one shot tasks
        dataloader: pytorch dataloader
        device: str, cuda of cpu
        verbose: bool, print percent correct or not

    Return: percent of correct one short tasks
    """
    n_correct = 0
    if verbose:
        print(
            "Evaluating model on {} random {} way one-shot learning tasks ... \n".format(
                k, N
            )
        )
    for i in range(k):
        img0, img1, targets = make_oneshot_task(N, dataloader)
        img0, img1, targets = img0.to(device), img1.to(device), targets.to(device)

        output = model(img0, img1)
        # take logits and look for the maximum
        prob_of_same = output[:, 1]
        
        # or take softmax, and all but the same images in the second column will have a small probability
        # prob = F.softmax(output, dim=1)
        # prob_of_same = prob[:, 1]
        if torch.argmax(prob_of_same) == torch.argmax(targets):
            n_correct += 1

    percent_correct = 100.0 * n_correct / k
    if verbose:
        print(
            "Got an average of {:.1f}% {} way one-shot learning accuracy \n".format(
                percent_correct, N
            )
        )
    return percent_correct


class AlbuSingleAttrDataset(Dataset):
    """ Create torch.Dataset based on dataframe with img path and its classes
    for the classification of a single attribute(column)
   
    """
    def __init__(self, label_df, target_attr, indexes, transform=None, path_col="path"):
        """Inits AlbuSingleAttrDataset
        
        Args:
            label_df: pd.Dataframe, with path_col and target_attr columns
            target_attr: str, name of classified attribute
            indexes: list of indexes (or special object)
            transform: albumentation image transforms
            path_col: str, columns name with image paths
        """

        label_df = label_df.loc[indexes, :].copy()
        self.categories = label_df.loc[:, target_attr]
        self.img_paths = label_df.loc[:, path_col]
        self.labels = label_df.loc[indexes, target_attr]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        file_path = self.img_paths.iloc[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if isinstance(self.labels.iloc[idx], np.ndarray):
            label = self.labels.iloc[idx]
        else:
            label = self.labels.iloc[idx].values
        # image is a np.array  with shape (1200, 867, 3) (height,width, channels)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        # after augmentation image is torch.Tensor with size [3, height, width]

        return image, label
    
    
class SiameseNetworkDataset(AlbuSingleAttrDataset):
    """ Create torch.Dataset based on dataframe with img path and its labels
        for siamese nets.
        
    For training a Siamese network, you need to inside one of the batch was balanced
    to the number of examples of the same class and different classes.

        Attributes:
            cat_to_im_paths - dict, mapping original label to set of img paths
            
    """

    def __init__(self, label_df, target_attr, indexes, transform=None, path_col="path"):
        """ Inits SiameseNetworkDataset
            
            Args:
                label_df: pd.Dataframe, with path_col and target_attr columns
                target_attr: str, name of classified attribute
                indexes: list of indexes (or special object)
                transform: albumentation image transforms
                path_col: str, columns name with image paths
        """
        super().__init__(
            label_df=label_df,
            target_attr=target_attr,
            indexes=indexes,
            transform=transform,
        )
        self.cat_to_im_paths = (
            label_df.groupby([target_attr])[path_col].apply(set).to_dict()
        )

    def get_image_names(self, idx):
        anchor_cat = self.categories.iloc[idx]
        img_0_name = self.img_paths.iloc[idx]
        is_same_class = random.randint(0, 1)  # np.random.binomial(1, p=0.5)
        if is_same_class:
            imgs_same_class = self.cat_to_im_paths[anchor_cat] - {img_0_name}
            if len(imgs_same_class) == 0:
                img_1_name = img_0_name
                # print("alarm, category = ", anchor_cat)
            else:
                img_1_name = np.random.choice(list(imgs_same_class))
        else:
            set_categories = self.cat_to_im_paths.keys() - {anchor_cat}
            negative_cat = np.random.choice(list(set_categories))
            imgs_negative_class = self.cat_to_im_paths[negative_cat]
            img_1_name = np.random.choice(list(imgs_negative_class))
        return img_0_name, img_1_name, is_same_class

    def __getitem__(self, idx):
        img_0_path, img_1_path, is_same_class = self.get_image_names(idx)
        im0 = get_image(img_0_path, self.transform)
        im1 = get_image(img_1_path, self.transform)

        # after augmentation image is torch.Tensor with size [3, height, width]
        return im0, im1, np.float32(is_same_class), img_0_path, img_1_path
    
    