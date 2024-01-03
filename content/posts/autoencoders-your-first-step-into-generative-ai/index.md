---
title: 'Autoencoders: Your First Step into Generative AI'
date: 2024-01-02T21:00:00+07:00
tags: ["deeplearning", "representation", "autoencoder"]
draft: false
description: "Generally, there are two popular basic variant of Generative AI: Autoencoders network and Generative Adversarial Network (GAN). In this series, we will discover the former one and leave the latter in another one."
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
twitter:
    images:
        - tw-cards.jpg
        - images/cover.jpg
cover:
    image: "cover.jpg" # image path/url
    alt: "Cover Post" # alt text
    caption: "Photo by [Alexandre Perotto](https://unsplash.com/@perotto) on [Unsplash](https://unsplash.com/photos/low-angle-photography-of-building-zCevd81eJDU)" # display caption under cover
    relative: true # when using page bundles set this to true
math: katex
keywords: ["deeplearning", "generative ai", "autoencoder", "data compress"]
summary: "Generally, there are two popular basic variant of Generative AI: Autoencoders network and Generative Adversarial Network (GAN). In this series, we will discover the former one and leave the latter in another one."
---


## Generative AI

When we discuss about generative models, most of us might be quickly triggered to imagine the greatness of current Large Language Models (LLMs) such as ChatGPT, Bard, Gemini, LLaMA, Mixtral, etc. Or instead, the Text-to-Image models like Dall-e and stable diffusion.

Basically, the generative model works by trying to produce or generate new data that similar into the sampled one. Generally, there are two popular basic variant of Generative AI: Autoencoders network and Generative Adversarial Network (GAN). In this series, we will discover the former one and leave the latter in another one.

## Autoencoder Network

![Autoencoder diagram (1).png](images/Autoencoder_diagram_(1).png)

Autoencoder is a neural network architecture that learn representation *(encode)* the input data, then tries to reconstruct the original data as close as possible by leveraging the learned representation *(encoded)* data. It is useful for denoising image, reduce dimensionality, or detecting outlier.

This network mainly consist of three parts: Encoder network, Decoder network, and Latent Space; each of them play an important role to make the model works. Let’s breakdown each of them below.

### Encoder Network

Encoder network is responsible to take an input data, then compress it into smaller representation of data (latent space). The layer size of this network usually shrinks as it closer to the last layer that produce the latent space. It is inline with our human intuition where *compress* means to make things smaller, and in this case, it is works by passing the input data to hidden layers which shrinks at each stage.

### Latent Space

Latent space is the encoder outputs, which we can also call it *learned representation*. It is actually the collection of vectors just like the input data, but in smaller dimension. In my opinion, the latent space has smaller dimension as it is only keep the important parts of the input data. These important parts were selected and evaluated in every step of *forward propagation* in the encoder network, hence producing such smaller representation.

This smaller representation later will be consumed as input for the decoder network.

### Decoder Network

In contrast to the encoder network, decoder network is responsible to reconstruct *(decode)* the learned representation to be as close as possible with the original input data. Instead of using the original input data, this network use the latent space as its input. It also means the decoder network forced to *generate* new data based on those representation (which hopefully representing the input data enough). Finally, in each training step the generated data will be compared to the original one until it is already close enough.

### Loss Function

As we mentioned earlier, the generated data needs to be compared with the original one to ensure it closeness. To do that, we need to define specific loss function which can be different for each field (e.g. image, text, audio). In this case, we will use image data for our discussion.

Now, to measure the closeness between generated and original images, we can employ MSE Loss below.


$$
MSE = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y_i})^2
$$

where:

$N:$ is the total samples

$y_i:$ is the original input data

$\hat{y_i}:$ is the reconstructed data

## Application of Autoencoder

1. **Dimensionality Reduction:** similar to PCA (Principal Component Analysis) which useful for visualization purpose.
2. **Feature Extraction:** generate the compressed latent representation to be used for the downstream tasks such as classification or prediction. The base model of BERT, GPT2, and its family is the examples of this application.
3. **File Compression:** by reducing the data dimensionality we are able to reconstruct data with smaller size (but also risk some data quality).
4. **Image Denoising:** remove the noise of the image that might be produced by high ISO or corrupted image. In order to do that, we must train the model to learn the difference between the clean image and the noisy one. Once trained, it expected able to reconstruct image with less noise.
5. **Image Generator:** able to generate or modify images that are similar to input data. It used the variant of autoencoder named Variational Autoencoder (VAE). It is also useful for data augmentation.

## Coding Time!

In this article, we will try to make our very first autoencoder. We will start with the simple one using Linear layers to compress the [EMNIST letter images](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip).

If you are prefer to jump ahead into the notebook, [please visit this link](https://colab.research.google.com/drive/1dfxufRQtSKVPgKTLleE4ylEtuVdsCRkn?usp=sharing).

### Dataset Preparation

First of all, let’s import and create config for our training purpose.

```python
# data manipulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# utils
import os
import gzip
import string
from tqdm.auto import tqdm
import time

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision import datasets

class config:
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    learning_rate = 1e-3
    log_step = 100
    seed=42
    latent_dim = 32
    inp_out_dim = 784
    hidden_dim = 128
```

When I explored the dataset, I found the original images were flipped and rotated like this.

![Rotated EMNIST images](images/Untitled.png#center)

Therefore, we need to fix it into correct direction. Fortunately, torchvision has very helpful utilities to perform data transformation.

```python
# transform data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(1),     # we need to flip and rotate cz the
    transforms.RandomRotation((90,90)),     # original img was flip and rotated.
    transforms.ToTensor()
])
```

There, we define 3 transformations.

1. **RandomHorizontalFlip:** flip image horizontally. This function require a parameter of probability of an image being flipped. Here we need all images to be flipped, therefore we fixed the probability to 1.
2. **RandomRotation:** rotate image by angle. This function require sequence number of angles, if we put only single number it will assume the rotation degree ranging from (-degrees, +degrees). Since we need all images to be rotated in the same direction, we fixed the degrees by feeding sequences (tuple) of 90.
3. **ToTensor:** convert images to tensor and scale it to range (0, 1) at the same time.

Now, lets download EMNIST data and put our defined transformation here. Don’t forget to set the splits into `letters` as we want to reconstruct letters data instead of numbers.

```python
# load EMNIST data
train_data = datasets.EMNIST(root='data', 
                             train=True, 
                             download=True, 
                             transform=transform,
                             split='letters')

test_data = datasets.EMNIST(root='data',
                            train=False,
                            download=True,
                            transform=transform,
                            split='letters')
```

With the transformation applied, now if we interpret our data it will be in the correct direction.

![Transformed EMNIST images](images/Untitled%201.png#center)

Next, let’s prepare our dataloaders!

```python
# setup the dataloaders
trainloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
testloader = DataLoader(test_data, shuffle=True, batch_size=config.batch_size)
```

### Designing Model Architecture

Now, we are close enough to the fun part. But before that, let’s build our model architecture first. Here we will use Linear Layer for both our Encoder and Decoder networks. Remember that our data is scaled within range (0,1). Therefore, we should put Sigmoid layer in the very last part of Decoder network.

```python
class LinearAutoencoder(nn.Module):
    def __init__(self, inp_out_dim, hidden_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        
        # encoder layer to latent space
        self.encoder = nn.Sequential(
            nn.Linear(inp_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # latent space to decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, inp_out_dim),
            nn.Sigmoid() # we use sigmoid cz the input and output should be in range 0,1
        )
        

    def forward(self, x):
        # pass input to encoder and activate it with ReLU
        x = self.encoder(x)
        # pass latent space to decoder and scale it with Sigmoid
        x = self.decoder(x)
        
        return x
```

Let’s define our model and interpret its architecture.

```python
# define model
model = LinearAutoencoder(inp_out_dim=config.inp_out_dim,
                          hidden_dim=config.hidden_dim,
                          latent_dim=config.latent_dim)

# move to GPU device
model = model.to(config.device)
print(model)
```

```
# our model architecture
LinearAutoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=32, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=32, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=784, bias=True)
    (3): Sigmoid()
  )
)
```

Take a look for a while in our model architecture.

Initially our model will accept the input image within the shape of (batch_size, 784). For those who are wondering why is it 784 and not other value, well, it is actually obtained from 28 * 28 which our original image size.

> **A little explanation…**
>
> For better intuition, let me break down a little bit for this part.
>
> By default, our data is arranged with the following shape format (batch_size, color_channel, height, width). If you take a batch from our trainloader, you will observe that our dataset having shape like this.
>
> `torch.Size([512, 1, 28, 28])`
>
> Then, we need to flatten it into 2-d array. So, later we will have dataset within shape (512, 784) which then will fed to our model.
> 

Now, back to our model architecture.

Instead of having single Linear Layer, we stack it with another hidden layer on each Encoder and Decoder network. You may try to modify the dimension of hidden layer by changing `hidden_dim` value in `config`.

Then, from hidden layer, we produce a latent representation within size dimension of 32. You also may modify it by changing `latent_dim` in `config`. Finally, the latent space will be act as the input of Decoder network.

> 💡 Note that the Encoder network should be shrinking to its end and the opposite for the Decoder network as our objective is to compress the images.

### Training Model

And we arrived to the most interesting part. Here we define our loss function (criterion), optimizer, and training function.

```python
# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# logging
history = {
    'train_loss': []
}

# progressbar
num_train_steps = len(trainloader) * config.epochs
progressbar = tqdm(range(num_train_steps))

# training function
epochtime = time.time()
for epoch in range(config.epochs):

    trainloss = 0
    batchtime = time.time()
    
    for idx, batch in enumerate(trainloader):
        # unpack data
        features, _ = batch
        features = features.to(config.device)

        # reshape input data into (batch_size x 784)
        features = features.view(features.size(0), -1)

        # clear gradient
        optimizer.zero_grad()

        # forward pass
        output = model(features)

        # calculate loss
        loss = criterion(output, features)
        loss.backward()

        # optimize
        optimizer.step()

        #  update running training loss
        trainloss += loss.item()

        # update progressbar
        progressbar.update(1)
        progressbar.set_postfix_str(f"Loss: {loss.item()}:.3f")

        # log step
        if idx % config.log_step == 0:
            print("Epoch: %03d/%03d | Batch: %04d/%04d | Loss: %.4f" \
                  % ((epoch+1), config.epochs, idx, \
                     len(trainloader), trainloss / (idx + 1)))
                    
    # log epoch
    history['train_loss'].append(trainloss / len(trainloader))
    print("***Epoch: %03d/%03d | Loss: %.3f" \
          % ((epoch+1), config.epochs, loss.item()))
    
    # log time
    print('Time elapsed: %.2f min' % ((time.time() - batchtime) / 60))
    
print('Total Training Time: %.2f min' % ((time.time() - epochtime) / 60))
```

Here we will train for 20 epochs in total, and we log our model performances to console for every 100 training steps.

Additionally, we can also plot our training history to get better understanding on model performance.

```python
plt.figure(figsize=(5, 7))
plt.plot(range(len(history['train_loss'])), history['train_loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
```

![Training Loss](images/Untitled%202.png#center)

After training for several epochs, we then evaluate it on our test set. Don’t forget to turn off the gradient by putting `torch.no_grad()` during evaluation since we don’t need any backpropagation process.

```python
# evaluate model
testloss = 0
testtime = time.time()

for batch in tqdm(testloader):
    # unpack data
    test_feats, _ = batch
    # reshape image
    test_feats = test_feats.view(test_feats.size(0), -1).to(config.device)
    # forward pass
    with torch.no_grad():
        test_out = model(test_feats)
    # compute loss
    loss = criterion(test_out, test_feats)
    testloss += loss.item()

print('Test Loss: %.4f' % (testloss / len(testloader)))
print('Total Testing Time: %.2f min' % ((time.time() - testtime) / 60))
```

![Test Loss](images/Untitled%203.png#center)

### Inference

It’s time to use our human intuition to see how good our model compression’s result. Let’s take a batch from the test set and compress it with our model.

```python
# obtain one batch of test images
test_feats, test_labels = next(iter(testloader))
original_img = test_feats.numpy()

# reshape image
test_feats = test_feats.view(test_feats.size(0), -1).to(config.device)

# forward pass
with torch.no_grad():
    infer_output = model(test_feats).detach().cpu()

# resize outputs back to batch of images
reconstructed_img = infer_output.view(config.batch_size, 1, 28, 28).numpy()
```

Finally, we will compare both original data and the compressed one.

```python
# plot the first ten input images and the reconstructed images
fig, axes = plt.subplots(2, 10, sharex=True, sharey=True, figsize=(25, 4))

# input images on top, reconstruction on bottom
for idx, (images, row) in enumerate(zip([original_img, reconstructed_img], axes)):
    for img, lbl, ax in zip(images, test_labels, row):
        ax.imshow(img.squeeze(), cmap=plt.cm.binary)
        if idx == 0:
            ax.set_title(f"Label: {alphabets[lbl-1]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
```

![Inference Samples](images/Untitled%204.png)

### Save Model

Lastly, if we are satisfied already with our model performance, we can save it. So we can use it anytime later without needing to run through all the codes above.

```python
torch.save(model.state_dict(), 'emnist-linear-autoencoder.pt')
```

## Conclusion

So we already discussed the Autoencoder network which also a family of Generative AI. It consists of 3 main parts: Encoder network, Decoder network, and the Latent representation. We also covered the implementation of Autoencoder using simple stacks of Linear Layer.

Although simple network, our model performs quite good on test set and able to compress and reconstruct letter images.

If you have any inquiries, comments, suggestions, or critics please don’t hesitate to reach me out:

- Mail: [affahrizain@gmail.com](mailto:affahrizain@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/fahrizainn/](https://www.linkedin.com/in/fahrizainn/)
- GitHub: [https://github.com/fhrzn](https://github.com/fhrzn)

Cheers!  🥂


---


## References

1. [https://www.analyticsvidhya.com/blog/2021/06/autoencoders-a-gentle-introduction/](https://www.analyticsvidhya.com/blog/2021/06/autoencoders-a-gentle-introduction/)
2. [https://structilmy.com/blog/2020/03/17/pengenalan-autoencoder-neural-network-untuk-kompresi-data/](https://structilmy.com/blog/2020/03/17/pengenalan-autoencoder-neural-network-untuk-kompresi-data/)
3. [https://medium.com/@samuelsena/pengenalan-deep-learning-part-6-deep-autoencoder-40d79e9c7866](https://medium.com/@samuelsena/pengenalan-deep-learning-part-6-deep-autoencoder-40d79e9c7866)
4. [https://deepai.org/machine-learning-glossary-and-terms/autoencoder](https://deepai.org/machine-learning-glossary-and-terms/autoencoder)
5. [https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder/linear-autoencoder](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder/linear-autoencoder)
6. [https://www.nist.gov/itl/products-and-services/emnist-dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
7. [https://www.youtube.com/watch?v=345wRyqKkQ0&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=138](https://www.youtube.com/watch?v=345wRyqKkQ0&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=138)
8. [https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L16](https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L16)