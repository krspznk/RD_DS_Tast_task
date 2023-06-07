Progress:
-Dataset Preparation:

*To get all files you should download data from Kaggle task and put all the data in a 'data'

-Exploratory Data Analysis:

*Perform exploratory data analysis on the dataset to gain insights into the distribution of ship and non-ship images, image sizes, and any other relevant statistics.
*It goes into jupyter notebook called 'exploratory_data_analysis' in notebook holder

-Load and preprocess the training data:
*Read the image files and their corresponding masks from the "train_v2" folder.
*Resize the images and masks to a consistent size, such as (768, 768).

-Model Architecture:

*Define the encoder part of the U-Net, which consists of convolutional and pooling layers to extract features from the input images.
*Define the decoder part of the U-Net, which consists of upsampling and concatenation layers to reconstruct the segmented output.
*Implement skip connections between the encoder and decoder to preserve spatial information.
*Combine the encoder, decoder, and skip connections to form the complete U-Net model.

-Dice Loss:

*The dice loss measures the overlap between the predicted segmentation and the ground truth segmentation.
*Calculate the dice coefficient as the intersection of the predicted and ground truth segments divided by the union of the two.

-Model Training:

* In train.py we can train model using train_v2 and train_ship_segmentations.csv and save model i folder 'models'

-Testing Model:

* In inference.py we can test model using trained_model_weights.h5 that we got from train.py and get a result in 'results' folder
