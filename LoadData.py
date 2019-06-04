
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import  matchingURL, mapFromWordToIdx,  CreateTweetsWithUserInfoDataset
import itertools
import pickle
from Constants import specialTokenList
import nltk
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import re


class filling_method():
    MOST_COMMON = "MOST_COMMON"
    MEAN = "MEAN"
    CERTAIN_VALUE = "CERTAIN_VALUE"


def TkloadingTweetsAndUserInfoData(args, resultTextbox, window):

    '''
    This function can load the data and perfrom the preprocessing
    '''
    

    # Check if the pre-processed datasets (training, validation, test) exist. if the pre-processed data already exist, then load it rather than perform pr-processing again.
    if not os.path.isfile(os.path.join(args.dataset, args.pickle_name)):
        
        # Check if the pre-processed df (overall Pandas dataframe). If it doesn't exist, load the original data to perform preprocessing.  
        if not os.path.isfile(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)):


            '''
            resultTextbox.insert("end", "String_Here") -> This function for adding a String to the result box
            window.update_idletasks() -> This function make the wondow to update the result box.
            '''

            # Adding the loading information to result box
            resultTextbox.insert("end", ("Loading " + str(os.path.join(
                args.dataset, "FullTweetsDataNoOrdered.html")) + ' and ' + str(os.path.join(
                    args.dataset, "FullExtraInfoDataNoOrdered.csv")) + " to do the Proprocessing\n"))
            window.update_idletasks()


            # Load the original data set, which contains a html (storing the tweets text) and a csv (storing other tweets' and users' information)

            # Load the tweets text
            tweets_df = pd.read_html(os.path.join(
                args.dataset, "FullTweetsDataNoOrdered.html"))
            tweets_df = pd.DataFrame(list(tweets_df[0].iloc[1:][0]))
            tweets_df.columns = ['text']

            # Load other information    
            extraInfo_df = pd.read_csv(os.path.join(
                args.dataset, "FullExtraInfoDataNoOrdered.csv"))
            


            # Concat two loaded dataframe 
            df = pd.concat([tweets_df, extraInfo_df], axis=1)

            # Delete the loaded dataframe after concating since we have the concatinated df now.
            del tweets_df
            del extraInfo_df


            # Show the dataset size in the result box
            resultTextbox.insert(
                "end", ("Dataset size: " + str(len(df)) + "\n"))
            window.update_idletasks()


            
            def preprocessingInputTextData(colName):

                '''
                This function is used for preprocessing
                '''
                input = df[colName]
                ps = nltk.stem.PorterStemmer() # Init Poter Stemmer
                tknzr = TweetTokenizer()   # Init Tweet Tokenizer
                allText = [i for i in input]     

                ## The detail preprocessing step is in the report
                preprocessedText = [[ps.stem(word) for word in tknzr.tokenize(re.sub(r'\d+', '', re.sub(r"http\S+|www.\S+", matchingURL,
                                                                                                        sentence)).lower()) if word not in nltk.corpus.stopwords.words('english') and len(word) >= 3] for sentence in allText]
                df[colName] = preprocessedText

            def fillingNullValue(colName):

                '''
                The function used for replace the 'nan' in the dataset
                '''
                
                if args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.MOST_COMMON:
                    ## replace the nan by mean
                    df[colName] = df[colName].astype('float')
                    df[colName].fillna(df[colName].mean(), inplace=True)
                elif args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.MEAN:
                    ## replace the nan by the most common value
                    df[colName] = df[colName].astype('category')
                    df[colName].fillna(df[colName].astype(
                        'category').describe()['top'], inplace=True)
                elif args.preprocessingStra[colName]['fillingNullMethod'] == filling_method.CERTAIN_VALUE:
                    ## replace the nan by a certain value
                    df[colName] = df[colName].astype('category')
                    df[colName] = df[colName].cat.add_categories(
                        [args.preprocessingStra[colName]['fillingNullValue']])
                    df[colName].fillna(args.preprocessingStra[colName]
                                       ['fillingNullValue'], inplace=True)

            def TweetsWithUserInfoPreprocessing():
                '''
                Perform the preprocessing here
                '''

                for colName in args.preprocessingStra.keys():
                    resultTextbox.insert(
                        "end", ("Preprocessing feature: " + str(colName) + "\n"))
                    window.update_idletasks()
                    for step in args.preprocessingStra[colName]['steps']:
                        if not step is None:
                            step(colName)
            ###############  Preprocessing Strategy ###############

            args.preprocessingStra = defaultdict(dict)
            args.preprocessingStra['text']['steps'] = [
                preprocessingInputTextData]
            args.preprocessingStra["numberOfHashtags_c"]['steps'] = [None]
            args.preprocessingStra['favorite_count']['steps'] = [None]
            args.preprocessingStra['retweet_count']['steps'] = [None]
            args.preprocessingStra['possibly_sensitive'] = {
                'fillingNullMethod': filling_method.CERTAIN_VALUE,
                'fillingNullValue': 'UNKNOWN',
                'steps': [fillingNullValue],
            }
            args.preprocessingStra['followers_count']['steps'] = [None]
            args.preprocessingStra['friends_count']['steps'] = [None]
            args.preprocessingStra['default_profile']['steps'] = [None]
            args.preprocessingStra['default_profile_image']['steps'] = [None]
            args.preprocessingStra['favourites_count']['steps'] = [None]
            args.preprocessingStra['listed_count']['steps'] = [None]
            args.preprocessingStra['statuses_count']['steps'] = [None]
            args.preprocessingStra['verified']['steps'] = [None]

            resultTextbox.insert("end", ('Preprocessing Strategy Set\n'))
            window.update_idletasks()

            #############################################################

            resultTextbox.insert("end", ('Start Preprocessing...\n'))
            window.update_idletasks()

            TweetsWithUserInfoPreprocessing()  # Apply inplace preprocessing

            # Get dummy variable
            df = pd.get_dummies(df, drop_first=True, columns=[
                                'possibly_sensitive', 'default_profile', 'default_profile_image', 'verified'])


            # Save the preprocessed-df
            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "wb") as fp:  
                pickle.dump(df, fp)
                resultTextbox.insert("end", ("The Pickle Data beforeMapToIdx Dumped to: " + str(os.path.join(
                    args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
                window.update_idletasks()

        else:
            # If the preprocessed-df exist, load it.
            print("Loading Existing BeforeMapToIdx file for Tweets and User: ",
                  os.path.join(args.dataset, args.pickle_name_beforeMapToIdx))

            resultTextbox.insert("end", ("Loading Existing BeforeMapToIdx file for Tweets and User: " +
                                         str(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx)) + "\n"))
            window.update_idletasks()
            with open(os.path.join(args.dataset, args.pickle_name_beforeMapToIdx), "rb") as fp:   
                df = pickle.load(fp)

        #################### After having the pre-processed df ####################

        resultTextbox.insert("end", ('Spliting Datasets...\n'))
        window.update_idletasks()

        ## split the df to training, validation and test set.

        if args.runningOnSmallDataset:
            # If this user want to test the program on the small dataset,
            # do a fake split to have a smaller dataset. X_temp (will be deleted later) will have 98% of data, and the small dataset size is 2%

            # Fake split
            X_temp, X_train, Y_temp, Y_train = train_test_split(df.drop(
                'maliciousMark', axis=1), df['maliciousMark'], test_size=0.02, stratify=df['maliciousMark'],  random_state=args.random_seed)
            # get real training and test set.    
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_train, Y_train, test_size=args.validation_portion, stratify=Y_train,  random_state=args.random_seed)

            # delete X_temp and Y_temp
            del X_temp
            del Y_temp
        else:
            # if not running on the small dataset, do normal data split.
            X_train, X_test, Y_train, Y_test = train_test_split(df.drop(
                'maliciousMark', axis=1), df['maliciousMark'], test_size=args.validation_portion, stratify=df['maliciousMark'],  random_state=args.random_seed)

        X_validation, X_test, Y_validation, Y_test = train_test_split(
            X_test, Y_test, test_size=args.test_portion, stratify=Y_test, random_state=args.random_seed)
        
        
        ## Show the datasets' sizes
        resultTextbox.insert("end", ("Dataset Size: " + str(len(X_train) + len(X_validation) + len(X_test))  + "\n"))
        resultTextbox.insert("end", ("TrainingSet Size: " + str(len(X_train)) + "\n"))
        resultTextbox.insert("end", ("ValidationSet Size: " + str(len(X_validation)) + "\n"))
        resultTextbox.insert("end", ("TestSet Size: " + str(len(X_test)) + "\n"))
        window.update_idletasks()

        resultTextbox.insert("end", ('Creating Tweets_text...\n'))
        window.update_idletasks()

        ## create nltk.Text, which will be used as a dictionray.
        tweets_text = nltk.Text(list(itertools.chain(*X_train['text'])))


        # check if the hyper-parameter vocab_size exists and filter out the low tf words.
        args.vocab_size = args.vocab_size or len(tweets_text.tokens)
        if args.vocab_size:  # and this if expression
            tweets_text.tokens = specialTokenList + \
                [w for w, _ in tweets_text.vocab().most_common(
                    args.vocab_size - len(specialTokenList))]
        else:
            tweets_text.tokens = specialTokenList + tweets_text.tokens
        args.vocab_size = len(tweets_text.tokens)  # change the vacab_size



        ## Map the terms to index for every dataset
        resultTextbox.insert("end", ('Maping Word To Idx: training set\n'))
        window.update_idletasks()
        X_train['text'] = mapFromWordToIdx(X_train['text'], tweets_text) 
        resultTextbox.insert("end", ('Maping Word To Idx: validation set\n'))
        window.update_idletasks()
        X_validation['text'] = mapFromWordToIdx(
            X_validation['text'], tweets_text)
        resultTextbox.insert("end", ('Maping Word To Idx: test set\n'))
        window.update_idletasks()
        X_test['text'] = mapFromWordToIdx(X_test['text'], tweets_text)

        resultTextbox.insert("end", ('Creating Torch Training Datasets...\n'))
        window.update_idletasks()


        # args.X_train = X_train
        # args.Y_train = Y_train

        # Create training, validation and test Pytorch dataset for feeding data into the Pytorch Neural Network
        # More datails are in the utils.py CreateTweetsWithUserInfoDataset function
        training_dataset = CreateTweetsWithUserInfoDataset(
            X_train, list(map(int, list(Y_train))))

        resultTextbox.insert(
            "end", ('Creating Torch Validation Datasets...\n'))
        window.update_idletasks()
        validation_dataset = CreateTweetsWithUserInfoDataset(
            X_validation, list(map(int, list(Y_validation))))

        resultTextbox.insert("end", ('Creating Torch Test Datasets...\n'))
        window.update_idletasks()
        test_dataset = CreateTweetsWithUserInfoDataset(
            X_test, list(map(int, list(Y_test))))

        resultTextbox.insert("end", ('Dumping data...\n'))
        window.update_idletasks()

        # Dump the pre-processed datasets
        with open(os.path.join(args.dataset, args.pickle_name), "wb") as fp:   # Pickling
            pickle.dump([training_dataset, validation_dataset,
                         test_dataset, tweets_text], fp)
            print("The Pickle Data Dumped to: ", os.path.join(
                args.dataset, args.pickle_name))
            resultTextbox.insert("end", ("The Pickle Data Dumped to: " + str(os.path.join(
                args.dataset, args.pickle_name)) + "\n"))
            window.update_idletasks()

    else:

        # If the pre-processed datasets exist, load it.
        resultTextbox.insert("end", ("Loading Existing File: " + str(os.path.join(
            args.dataset, args.pickle_name)) + '\n'))
        window.update_idletasks()

        with open(os.path.join(args.dataset, args.pickle_name), "rb") as fp:   # Unpickling
            training_dataset, validation_dataset, test_dataset, tweets_text = pickle.load(
                fp)


    ## Some dataframe hyper-parameters that will be used latter
    args.vocab_size = len(tweets_text.tokens)
    args.num_extra_info = len(training_dataset[0][1])
    args.num_features = len(training_dataset[0][1]) + 1


    # return the loaded or generated dataset.
    return training_dataset, validation_dataset, test_dataset, tweets_text
