import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from src_other import utils


class FF_senti(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.senti = utils.get_senti_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        print("senti initialized")

    def __getitem__(self, index):
        # print("get_item reached")
        # sample, class_label = self._generate_sample(
        #     index
        # )
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }

        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.senti[1])

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            class_label.clone().detach(), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample = torch.concat((one_hot_label, pos_sample), dim=0)
        # pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        wrong_class_label = 1 - class_label
        one_hot_label = torch.nn.functional.one_hot(
            wrong_class_label.clone().detach(), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample = torch.concat((one_hot_label, neg_sample), dim=0)
        return neg_sample

    def _get_neutral_sample(self, z):
        return torch.concat((self.uniform_label, z), dim=0)
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0]+self.num_classes))
        for i in range(self.num_classes):
            all_samples[i, self.num_classes:] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.senti[0][index], self.senti[1][index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label


class FF_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_CIFAR10_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        # print(pos_sample.shape, one_hot_label.shape)
        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label

class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()

        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label
    


class FF_CIFAR100(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=100):
        self.opt = opt
        self.cifar100 = utils.get_CIFAR100_partition(opt, partition)  # Use CIFAR-100 loader
        self.num_classes = num_classes
        self.num_classes_max = 32
        self.uniform_label = torch.ones(self.num_classes_max) / self.num_classes_max

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.cifar100)

    def _get_pos_sample(self, sample, class_label):
        h,w =self._get_width_height(class_label)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(w), num_classes=self.num_classes_max
        )
        pos_sample = sample.clone()
        # print(pos_sample.shape, one_hot_label.shape)
        pos_sample[0,h, : self.num_classes_max] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        h,w =self._get_width_height(class_label)
        classes = list(range(self.num_classes_max))
        classes.remove(w)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes_max
        )
        neg_sample = sample.clone()
        neg_sample[0, w, : self.num_classes_max] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0:3, : self.num_classes] = self.uniform_label
        return z
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            h,w =self._get_width_height(i)
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(w), num_classes=self.num_classes_max)
            all_samples[i, 0, h, : self.num_classes_max] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get CIFAR-100 sample.
        sample, class_label = self.cifar100[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label
    
    def _get_width_height(self, class_label):
        h = class_label//32
        w = class_label%32
        return h, w
        

class CwC_CIFAR(torch.utils.data.Dataset):
    def __init__(self, opt, partition, number_samples = None, size=(32, 32), dataset='cifar10',  increment_dif=False, FF_rep=False):
        
        if number_samples is None:
            if partition in ["train"]:
                print("Using 50000 samples for training")
                self.number_samples = 50000
            else:
                print("Using 10000 samples for validation")
                self.number_samples = 10000
        else:    
            self.number_samples = number_samples

        if dataset == 'cifar10':
            self.data_samples = utils.get_CIFAR10_partition(opt, partition)
        else:
            self.data_samples = utils.get_CIFAR100_partition(opt, partition)

        self.size = size
        self.increment_dif = increment_dif
        self.normal_imgs, self.y_pos = self.generate_data()
        self.FF_rep = FF_rep
        self.dataset = dataset

        

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2


    def visualize(self, digit1, digit2, mask, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
        ax1.imshow(digit1, cmap='gray')
        ax1.set_title('Digit 1')
        ax2.imshow(digit2, cmap='gray')
        ax2.set_title('Digit 2')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('Binary Mask')
        ax4.imshow(hybrid, cmap='gray')
        ax4.set_title('Hybrid 1')
        plt.show()

        return

    def generate_data(self):

        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):

            idx1 = i
            y = self.data_samples[idx1][1]
            digit1 = np.array(self.data_samples[idx1][0])
            normal_imgs.append(digit1)
            y_pos.append(y)

        return normal_imgs, y_pos

    def __getitem__(self, index):

        # CIFAR100 experiment.
        # Following standard practice, normalized the channels by ((0.5074,0.4867,0.4411) and  (0.2011,0.1987,0.2025)
        # if self.dataset == 'CIFAR100':
        #     transform = Compose([
        #         ToTensor(),
        #         Normalize(mean=[0.5074, 0.4867, 0.4411],
        #                              std=[0.2011, 0.1987, 0.2025])])
        # else:
        #     transform = Compose([
        #         ToTensor(),
        #         Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])])

        # # Load Positive and Negative Samples
        # x_pos = transform(self.normal_imgs[index])
        x_pos = self.normal_imgs[index]
        y_pos = self.y_pos[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(3, self.size[1], self.size[0])  #
        y_pos = torch.tensor(np.asarray(y_pos)).long()

        if self.FF_rep:
            x_pos = x_pos.flatten()

        
        return x_pos, y_pos

    def __len__(self):
        return len(self.normal_imgs)


class CwC_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, number_samples = None,  size=(28, 28), dataset='MNIST',  increment_dif=False, FF_rep=False):
        
        if number_samples is None:
            if partition in ["train"]:
                print("Using 50000 samples for training")
                self.number_samples = 50000
            else:
                print("Using 10000 samples for validation")
                self.number_samples = 10000
        else:
            self.number_samples = number_samples
        

        self.data_samples = utils.get_MNIST_partition(opt, partition)
        self.normal_imgs, self.y_pos = self.generate_data()
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.increment_dif = increment_dif
        self.dataset = dataset
        self.FF_rep = FF_rep

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2


    def visualize(self, digit1, digit2, mask, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
        ax1.imshow(digit1, cmap='gray')
        ax1.set_title('Digit 1')
        ax2.imshow(digit2, cmap='gray')
        ax2.set_title('Digit 2')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('Binary Mask')
        ax4.imshow(hybrid, cmap='gray')
        ax4.set_title('Hybrid 1')
        plt.show()

        return

    def generate_data(self):

        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):

            idx1 = i

            y = self.data_samples[idx1][1]

            digit1 = np.array(self.data_samples[idx1][0])

            normal_imgs.append(digit1)
            y_pos.append(y)

        return normal_imgs, y_pos

    def __getitem__(self, index):

        

        # Load Positive and Negative Samples
        x_pos = self.normal_imgs[index]
        y_pos = self.y_pos[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(1, self.size[1], self.size[0])  #

        y_pos = torch.tensor(np.asarray(y_pos)).long()

        if self.FF_rep:
            x_pos = x_pos.flatten()

        return x_pos, y_pos

    def __len__(self):
        return len(self.normal_imgs)
