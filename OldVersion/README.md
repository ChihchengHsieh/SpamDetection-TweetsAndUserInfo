


# Instructions: 

1. Clone this folder and extract it.

2. Download the [[Dataset](https://drive.google.com/file/d/127FBr3Zs7rhGT07DFohNHY6gE1MKUhrJ/view?usp=sharing)] and put it inside the folder 

so the strcuture of this folder will become:

![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/Instructions/folderStructure.png?raw=true)

3. Then in the bash to run this command:

![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/Instructions/commandRun.png?raw=true)

4. The Tkinter Interface will pop out and you can train and test the models on it.

# Features in the dataset:



### Tweets
	text
	numberOfHashtags_c
	favorite_count
	retweet_count
	possibly_sensitive

### User:
	followers_count 
	friends_count
	default_profile 
	default_profile_image
	favourites_count
	listed_count
	statuses_count
	verified




# Model Structure

![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/Instructions/NewSpamDetectionModel.png?raw=true)


# Update for the small dataset running test:

![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/runningOnSamllDataset.png)

As we see from the above image, a samll check button (runningOnSamllDataset) has been added above the result box. If this check button is ticked. The program will run on a small dataset to check if the preprocessing and environment is good to go. The samll dataset size is 22903 only. So it should be ab proper size for both cpu and gpu users. 


# Results
## Using Both Models
### SSCL
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/SSCL_MultiTask.png?raw=true)

### GatedCNN
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/GatedCNN_MultiTask.png?raw=true)

### SelfAttn
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/SelfAttn_MultiTask.png?raw=true)

## Only textModel

### SSCL
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/textModelOnly_SSCL.png?raw=true)

### GatedCNN
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/textModelOnly_GatedCNN.png.png?raw=true)

### SelfAttn
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/textModelOnly_SelfAttn.png?raw=true)

## Only infoModel
![](https://github.com/ChihchengHsieh/SpamDetection-TweetsAndUserInfo/blob/master/TrainingResult/OnlyInfoModel.png?raw=true)
