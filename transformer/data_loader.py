import csv

from sklearn.model_selection import train_test_split

from torchnlp.datasets.dataset import Dataset


def twitter_airline_dataset(dev_split: float = 0.1,
                 test_split: float = 0.1,
                 directory: str = 'data',
                 name_file: str= 'Tweets.csv',
                 brand: list = [],
                 sentiments: str = ['neutral', 'positive', 'negative'],
                 sentiment_confidence_th: float = 0.5, 
                 random_seed: int = 3
                 ) -> (Dataset, Dataset, Dataset):
    """
    Load the Twitter US Airline Sentiment dataset.

    **Reference:** https://www.kaggle.com/crowdflower/twitter-airline-sentiment#Tweets.csv

    :param dev_split: percentage of the dataset to be used for development.
    :param test_split: percentage of the dataset to be used for testing.
    :param directory: Directory where the dataset is located.
    :param name_file: Name of the csv file. 

    :param brand (optional): list specifying the airlines to include in the dataset. If not specified, the list is empty and all the brands are considered. 
    :param sentiments (optional): Sentiments to load from the dataset.
    :param sentiment_confidence_th (optional): Sentiment confidence threshold.  
    :param random_seed (optional): seed used to split the corpus.
    """

    examples = []
    with open(directory+'/'+name_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)

        for line in csv_reader:
            
            target = line[header.index('airline_sentiment')]
            airline_sentiment_conf = float(line[header.index('airline_sentiment_confidence')])
            airline = line[header.index('airline')]
            source = line[header.index('text')]

            #select the data that fulfills the selected parameters 
            if (target in sentiments) and (airline_sentiment_conf >= sentiment_confidence_th and airline_sentiment_conf<=1) and \
                 (airline in brand if len(brand) != 0 else True) and (source != ''):
                examples.append({'source': source, 'target': target})

    dataset_size = len(examples)
    train, dev = train_test_split(examples, test_size=int(dataset_size*dev_split), random_state=random_seed)

    train, test = train_test_split(train, test_size=int(dataset_size*test_split), random_state=random_seed)
    
    return Dataset(train), Dataset(dev), Dataset(test)
