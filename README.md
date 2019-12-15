# indoor-scenes
Using cnn to classify scenes like (airport inside, bakery, restaurant, gym, green house, kitchen, pool inside ,toy store, operating room, bedroom), our [Data](https://www.kaggle.com/c/fcis-cs-deeplearningcompetition/data) was imblaced so we made image augmentation(fliping, random crop and add noise) after that we used pretrained vgg19 as feature extractor then train extracted features with two FC layers.
we got accuracy 92% on test data and 99% on train data.    
this images classified by the model..
 <img src="/classification.png" width="1000" height="1000">
