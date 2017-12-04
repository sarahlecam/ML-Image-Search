import pandas as pd
import csv
import glob
import re
import numpy as np
import scipy as sp
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn import preprocessing as pp
from sklearn.neighbors import KNeighborsClassifier 


# parse features and labels for train and test data
def parse_data(fpath):
    df = pd.read_csv(fpath, header=None)
    features = df.iloc[:,1:]
    labels = df.iloc[:,0]
    return np.array(features), np.array(labels)

train_feat_path = 'data/features_train/features_resnet1000_train.csv'
test_feat_path = 'data/features_test/features_resnet1000_test.csv'

train_features, train_labels = parse_data(train_feat_path)
test_features, test_labels = parse_data(test_feat_path)


# preprocess descriptions to remove noises
def preprocess_descriptions(fpath):
    stemmer = PorterStemmer()
    descriptions = []
    
    for fname in glob.glob(fpath):
        file = open(fname, 'r')
        desc = file.read()
        desc = np.char.lower(desc)

        # replace punctuations in each set of descriptions
        desc = re.sub('[^\w\s]', '' , str(desc))
        
        words = []
        for i, word in enumerate(desc.split()):
            if word not in stopwords.words('english'):
                # if not stopword, lemmatize 
                words.append(stemmer.stem(word))
        description = ' '.join(words)
        image_idx = int(fname.split('/')[-1].split('.')[0])
        descriptions.insert(image_idx, description)
    return descriptions

train_desc_fpath = 'data/descriptions_train/*.txt'
test_desc_fpath = 'data/descriptions_test/*.txt'

train_desc = preprocess_descriptions(train_desc_fpath)
test_desc = preprocess_descriptions(test_desc_fpath)


# build bag of words for descriptions
def build_bag_of_words(descriptions):
    unique_words = []
    desc_feat = []
    
    #build a list of unique words from train descriptions
    for desc in train_desc:
        for word in desc.split():
            if word not in unique_words:
                unique_words.append(word)

    # for each description, create a feature vector with ith index as the count of ith word in the word list
    for desc in descriptions:
        feat = [0] * len(unique_words) 
        for word in desc.split():
            if word in unique_words:
                feat[unique_words.index(word)] += 1
        desc_feat.append(feat)

    #print(unique_words)
    return np.array(desc_feat)

train_desc_feat, test_desc_feat = [], []
train_desc_feat = build_bag_of_words(train_desc)
test_desc_feat = build_bag_of_words(test_desc)


def post_process_descriptions(desc_feat):
    desc_feat = pp.normalize(desc_feat, norm='l2')
    return desc_feat

train_desc_feat = post_process_descriptions(train_desc_feat)
test_desc_feat = post_process_descriptions(test_desc_feat)


# train KNeighborsClassifier to fit training description feature to their image labels
def train_KNN():
    knn = KNeighborsClassifier()
    knn.fit(train_desc_feat, np.array(range(10000)))
    # find closest training image for test description features
    pred = knn.predict(test_desc_feat)
    return pred

pred = train_KNN()


# create map of image name to train feature vector
def create_train_features_map():
	image_features_dict = {}
	for i, feat in enumerate(train_features):
		image_name = train_labels[i]
		image_idx = int(image_name.split('/')[1].split('.')[0])
		image_features_dict[image_idx] = feat
	return image_features_dict

image_features_dict = create_train_features_map()


def get_dist(feat1, feat2):
    dist = sp.spatial.distance.cdist(feat1.reshape(1, -1), feat2.reshape(1, -1), 'euclidean').flatten()
    return dist


def get_top_20_images(desc_idx):
    top_images = []
    pred = predictions[desc_idx]
    # get training feature vector for predicted training image
    train_feat = image_features_dict[pred]
    feat_score = {} # map of test label to dist with train features
    for i, test_feat in enumerate(test_features):
        # compare test feature with train feature
        score = get_dist(train_feat, test_feat)
        feat_score[test_labels[i]] = score
    #print(feat_score)
    sorted_feat_score = sorted(feat_score.items(), key=operator.itemgetter(1))
    for k,v in sorted_feat_score[:20]:
        top_images.append(k.split('/')[1])
    return top_images


def write_submissions(output_filename):
    output_file = open(output_filename, "w")
    writer = csv.writer(output_file)
    #write headers
    writer.writerow(["Descritpion_ID", "Top_20_Image_IDs"])
    # get top 20 images for each test description
    for i in range(len(test_desc)):
        top_20_images = get_top_20_images(i)
        images = " ".join(top_20_images)
        writer.writerow([str(i) + '.txt', images])


write_submissions('sample_submission.csv')
