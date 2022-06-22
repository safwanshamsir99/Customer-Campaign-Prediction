<a><img alt = 'python' src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'spyder' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
<a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
![customer campaign](static/customer_campaign.jpg)

# Outcome Prediction Using Deep Learning Model for Customer Campaign.
 Trained over 31,000 data to predict the outcome of the term deposit subscription by customer based on a campaign.

## Description
1. The project's objective is to predict the chance the outcome of a campaign based on the customer's data
2. Based on the dataset description, the dataset contains details of marketing campaigns done via phonecall with various details for customers such as demographics, last campaign details, many more.
3. The dataset contains 6 continuous features, 10 categorical features, 1 categorical target. It has no duplicate data but has a lot of NaNs especially for 'days_since_prev_campaign_contact' column. The dataset can be downloaded from the link given in the credit section below.
4. The only features selected from the dataset are all of the continuous columns since they have the highest correlation to the target, which is outcome of the campaign.
5. By using the simple two layer deep learning model which only comprises of Dense, Dropout, and Batch Normalization layers, the model successfully achieved 90.1% accuracy.
6. Methods that used to improve the deep learning model are by increasing the number of epochs inside the model and by using the early stopping callbacks to prevent the model from overfitting. 

### Deep learning model image
![model_score](static/model.png)

## Results
Training loss & Validation loss:
![model_loss](static/loss_campaign.png)

Training accuracy & Validation accuracy:
![model_accuracy](static/accuracy_campaign.png)

Model score:
![model_score](static/model_score.png)

## Discussion
1. The deep learning model achieved 90.1% accuracy during the model evaluation. 
2. Recall and f1 score reported 99% and 95% respectively. 
3. Early stopping callback was used to prevent overfitting of the deep learning model, and during the model training, the epochs stop at 32.
4. Tensorboard was used to visualize the the loss and accuracy graph of the model. The snipping shot from the tensorboard is shown below.
![tensorboard](static/tensorboard_campaign.png.png)

## Credits:
Shout out to Kunal Gupta from Kaggle for the Customer Segmentation Dataset. Check out the dataset by clicking the link below. :smile:
### Dataset link
[HackerEarth HackLive: Customer Segmentation](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)
