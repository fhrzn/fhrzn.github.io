---
title: 'Exploring Vision Transformers (ViT) with 🤗 Huggingface'
date: 2022-10-14T23:53:18+07:00
tags: ["deeplearning", "computervision"]
draft: false
description: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)"
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
cover:
    image: "https://unsplash.com/photos/MAYsdoYpGuk/download?ixid=M3wxMjA3fDB8MXxhbGx8M3x8fHx8fDJ8fDE3MDM3ODgzODd8&force=true&w=640" # image path/url
    alt: "Cover Post" # alt text
    caption: "Photo by [Alex Litvin](https://unsplash.com/@alexlitvin?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)" # display caption under cover
    relative: true # when using page bundles set this to true
---

Lately, I was working on a course project where we asked to review one of the modern DL papers from top latest conferences and make an experimental test with our own dataset. So, here I am thrilled to share with you about my exploration!

## Background

As self-attention based model like Transformers has successfully become a _standard_ in NLP area, it triggers researchers to adapt attention-based models in Computer Vision too. There were different evidences, such as combine CNN with self-attention and completely replace Convolutions. While this selected paper belongs to the latter aproach.

The application of attention mechanism in images requires each pixel attends to every other pixel, which indeed requires expensive computation. Hence, several techniques have been applied such as self-attention only in local neighborhoods [1], using local multihead dot product self-attention blocks to completely replace convolutions [2][3][4], postprocessing CNN outputs using self- attention [5][6], etc. Although shown promising results, these techniques quite hard to be scaled and requires complex engineering to be implemented efficiently on hardware accelerators.

On the other hand, Transformers model is based on MLP networks, it has more computational efficiency and scalability, making its possible to train big models with over 100B parameters.

## Methods

![ViT Architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/1*-HQPfbnebarylP543i58_Q.png)
*General architecture of ViT. Taken from the original paper (Dosovitskiy et al., 2021)*

The original Transformers model treat its input as sequences which very different approach with CNN, hence the inputted images need to be extracted into fixed-size patches and flattened. Similar to BERT [CLS] token, the so-called _classification token_ will be added into the beginning of the sequences, which will serve as image representation and later will be fed into classification head. Finally, to retain the positional information of the sequences, positional embedding will be added to each patch.

The authors designed model following the original Transformers as close as possible. The proposed model then called as Vision Transfomers (ViT).

## Experiments

The authors released 3 variants of ViT; ViT-Base, ViT-Large, and ViT-Huge with different number of layers, hidden layers, MLP size, attention heads, and number of params. All of these are pretrained on large dataset such as ImageNet, ImageNet-21k, and JFT.

In the original paper, the author compared ViT with ResNet based models like BiT. The result shows ViT outperform ResNet based models while taking less computational resources to pretrain.

The following section will become technical part where we will use 🤗 Huggingface implementation of ViT to finetune our selected dataset.


## 🤗 Huggingface in Action

Now, let’s do interesting part. Here we will finetune ViT-Base using [Shoe vs Sandal vs Boot dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images) publicly available in Kaggle and examine its performance.

First, lets load the dataset using 🤗 Datasets.

```python3
from torch.utils.data import DataLoader  
from datasets import load_datasetdatasets = load_dataset('imagefolder', data_dir='../input/shoe-vs-sandal-vs-boot-dataset-15k-images/Shoe vs Sandal vs Boot Dataset')datasets_split = datasets['train'].train_test_split(test_size=.2, seed=42)  
datasets['train'] = datasets_split['train']  
datasets['validation'] = datasets_split['test']
```

Lets examine some of our dataset

```python3
# plot samples  
samples = datasets['train'].select(range(6))  
pointer = 0  
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,6))for i in range(2):  
    for j in range(3):  
        ax[i,j].imshow(samples[pointer]['image'])  
        ax[i,j].set_title(f"{labels[samples[pointer]['label']]} ({samples[pointer]['label']})")  
        ax[i,j].axis('off')  
        pointer+=1
        plt.show()
```

![Dataset sneak peek](https://miro.medium.com/v2/resize:fit:640/format:webp/1*c4RvCzuh84nsUw5kn28PqA.png#center)
*Few of our dataset looks like*

Next, as we already know, we need to transform our images into fixed-size patches and flatten it. We also need to add positional encoding and the _classification token._ Here we will use 🤗 Huggingface Feature Extractor module which do all mechanism for us!

This Feature Extractor is just like Tokenizer in NLP. Let’s now import the pretrained ViT and use it as Feature Extractor, then we will examine the outputs of processed image. Here we will use pretrained ViT with `patch_size=16` and pretrained on ImageNet21K dataset with resolution 224x224.

```python3
model_ckpt = 'google/vit-base-patch16-224-in21k'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
extractor = ViTFeatureExtractor.from_pretrained(model_ckpt)extractor(samples[0]['image'], return_tensors='pt')
```

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*zrS0kcR2kBmOtLpFCuHa2Q.png#center)
*Our extracted features looks like*

Note that our original image has white background, that’s why our extracted features having a lot of `1.` value. Don’t worry, its normal, everything will be work :)

Let’s proceed to the next step. Now we want to implement this feature extractor to the whole of our dataset. Generally, we could use `.map()` function from 🤗 Huggingface, but in this case it would be slow and time consuming. Instead, we will use `.with_transform()` function which will do transformation on the fly!

```python3
def batch_transform(examples):  
    # take a list of PIL images and turn into pixel values  
    inputs = extractor([x for x in examples['image']], return_tensors='pt')  
    # add the labels in  
    inputs['label'] = examples['label']  
      
    return inputstransformed_data = datasets.with_transform(batch_transform)
```

OK, so far we’re good. Next, let’s define our data collator function and evaluation metrics.

```python3
# data collator  
def collate_fn(examples):  
    return {  
        'pixel_values': torch.stack([x['pixel_values'] for x in examples]),  
        'labels': torch.tensor([x['label'] for x in examples])  
    }# metrics  
metric = load_metric('accuracy')  
def compute_metrics(p):  
    labels = p.label_ids  
    preds = p.predictions.argmax(-1)  
    acc = accuracy_score(labels, preds)  
    f1 = f1_score(labels, preds, average='weighted')  
    return {  
        'accuracy': acc,  
        'f1': f1  
    }
```

Now, let’s load the model. Remember that we have 3 labels in our data, and we attach it as our model parameters, so we will have ViT with classification head output of 3.

```python3
model = ViTForImageClassification.from_pretrained(  
    model_ckpt,  
    num_labels=len(labels),  
    id2label={str(i): c for i, c in enumerate(labels)},  
    label2id={c: str(i) for i, c in enumerate(labels)}  
)  
model = model.to(device)
```

Let’s have some fun before we finetune our model! (This step is optional, if you want to jump into fine-tuning step, you can skip this section).

I am quite interested to see ViT performance in zero-shot scenario. In case you are unfamiliar with _zero-shot_ term, it just barely use pretrained model to predict our new images. Keep in mind that most of pretrained model are trained on large datasets, so in _zero-shot_ scenario we want to take benefit from those large dataset for our model to identify features in another image that might haven’t see it before and then make a prediction. Let’s just see how it works in the code!

```python3
# get our transformed dataset  
zero_loader = DataLoader(transformed_data['test'], batch_size=16)  
zero_pred = []# zero-shot prediction  
for batch in tqdm(zero_loader):  
    with torch.no_grad():  
        logits = model(batch['pixel_values'].to(device)).logits  
        pred = logits.argmax(-1).cpu().detach().tolist()  
        zero_pred += [labels[i] for i in pred]zero_true = [labels[i] for i in datasets['test']['label']]# plot confusion matrix  
cm = confusion_matrix(zero_true, zero_pred, labels=labels)  
disp = ConfusionMatrixDisplay(cm, display_labels=labels)  
disp.plot()  
plt.show()# metrics  
print(f'Acc: {accuracy_score(zero_true, zero_pred):.3f}')  
print(f'F1: {f1_score(zero_true, zero_pred, average="weighted"):.3f}')
```

In short, we put our transformed data in DataLoader which going to be transformed on the fly. Then, for every batch, we pass our transformed data into our pretrained model. Next, we take the logits only from the model output. Remember that we have classification head with number of output 3. So, for each inferred image we will have 3 logits score. Among these 3 score, we will take the maximum one and return its index using `.argmax()`. Finally, we plot our confusion matrix and print the accuracy and F1 score.

![confusion matrix](https://miro.medium.com/v2/resize:fit:640/format:webp/1*o0KeIxC7nfv3-v43EBqPDA.png#center)
*ViT confusion matrix on zero-shot scenario*

Surprisingly, we got a unsatisfied metrics score with `Accuracy: 0.329` and `F1-Score: 0.307`. OK, next let’s fine-tune our model for 3 epochs and test the performance again. Here, I used Kaggle environment to train model.

```python3
batch_size = 16  
logging_steps = len(transformed_data['train']) // batch_sizetraining_args = TrainingArguments(  
    output_dir='./kaggle/working/',  
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size,  
    evaluation_strategy='epoch',  
    save_strategy='epoch',  
    num_train_epochs=3,  
    fp16=True if torch.cuda.is_available() else False,  
    logging_steps=logging_steps,  
    learning_rate=1e-5,  
    save_total_limit=2,  
    remove_unused_columns=False,  
    push_to_hub=False,  
    load_best_model_at_end=True)trainer = Trainer(  
    model=model,  
    args=training_args,  
    data_collator=collate_fn,  
    compute_metrics=compute_metrics,  
    train_dataset=transformed_data['train'],  
    eval_dataset=transformed_data['validation'],  
    tokenizer=extractor)trainer.train()
```

The code above was responsible to train our model. Note that we used 🤗 Huggingface Trainer instead of write our own training loop. Next, lets examine our Loss, Accuracy, and F1 Score for each epochs. You can also specify WandB or Tensorboard in Trainer parameter `report_to` for better logging interface. (Honestly, here I am using wandb for logging purpose. But for simplicity, I skipped the explanation of wandb part)

![model performances](https://miro.medium.com/v2/resize:fit:720/format:webp/1*P_yuwU4yPELwlUV8Xb1EcA.png)
*Model performances on each epochs*

Impressive, isn’t it? Our ViT model already got very high performance since the first epoch, and changing quite steadily! Finally, let’s test again on the test data and later we plot our model prediction on few of our test data.

```python3
# inference on test data  
predictions = trainer.predict(transformed_data['test'])  
predictions.metrics  
# plot samples  
samples = datasets['test'].select(range(6))  
pointer = 0fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10,6))  
for i in range(2):  
    for j in range(3):  
        ax[i,j].imshow(samples[pointer]['image'])  
        ax[i,j].set_title(f"A: {labels[samples[pointer]['label']]}nP: {labels[predictions.label_ids[pointer]]}")  
        ax[i,j].axis('off')  
        pointer+=1plt.show()
```

Here is our prediction scores on test data. Our finetuned model now has a very good performances compared to the one in _zero-shot_ scenario. And among of 6 sampled test images, our model correctly predict all of them. Super! ✨

```
{'test_loss': 0.04060511291027069,    
 'test_accuracy': 0.994,    
 'test_f1': 0.9939998484491527,    
 'test_runtime': 30.7084,    
 'test_samples_per_second': 97.693,    
 'test_steps_per_second': 6.122}
```

![prediction result](https://miro.medium.com/v2/resize:fit:640/format:webp/1*FCx445gVXRtjQ69YXVECbQ.png#center)

*Prediction result*

## Conclusion

Finally, we reached the end of the article. To recap, we did quick review of the original paper of Vision Transformers (ViT). We also perform _zero-shot_ and finetuning scenario to our pretrained model using publicly available Kaggle Shoe vs Sandals vs Boots dataset containing ~15K images. We examined that ViT performance on _zero-shot_ scenario wasn’t so good, while after finetuning the performance boost up since the first epoch and changing steadily.

If you found this article is useful, please don’t forget to clap and follow me for more Data Science / Machine Learning contents. Also, if you found something wrong or interesting, please feel free to drop it in the comment or reach me out at Twitter or Linkedin.

In case you are interested to read more, follow our medium [Data Folks Indonesia](https://medium.com/data-folks-indonesia) and don’t forget join us [Jakarata AI Research on Discord](https://discord.com/invite/6v28dq8dRE)!

Full codes are available on my [Github repository](https://github.com/fhrzn/sml-tech/blob/main/Tasks/Course%20Project/vit-shoe-vs-sandals.ipynb), feel free to check it 🤗.

_NB: If you are looking for deeper explanation especially if you want to reproduce the paper by yourself, you can check this_ [_amazing article by Aman Arora_](https://amaarora.github.io/2021/01/18/ViT.html)_._

---

## References

1.  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
2.  Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In _ICML_, 2018.
3.  Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In _ICCV_, 2019.
4.  Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. In _NeurIPS_, 2019.
5.  Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In _CVPR_, 2020.
6.  Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In _CVPR_, 2018.
7.  Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _ECCV_, 2020.

## Let’s Get in Touch!

*   [Linkedin](https://www.linkedin.com/in/fahrizainn/)
*   [Twitter](https://twitter.com/fhrzn_)
*   [Github](https://github.com/fhrzn)
