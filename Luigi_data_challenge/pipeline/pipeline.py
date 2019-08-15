""" Rubikloud take home problem """
import pandas as pd
import numpy as np
import math
import luigi

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.externals import joblib



class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        print('CleanData output')
        return luigi.LocalTarget(self.output_file)

    def run(self):
        print('CleanData run')
        tweets_raw = pd.read_csv(self.tweet_file, encoding = 'unicode_escape')
        tweets_df = tweets_raw[['airline_sentiment','tweet_coord']]
        # dropping NA
        tweets_df = tweets_df[tweets_df['tweet_coord'].notnull()]
        # dropping '[0.0, 0.0]'
        tweets_df = tweets_df.loc[tweets_df['tweet_coord'] != '[0.0, 0.0]']
        # saving to file
        with self.output().open('w') as outfile:
            print('CleanData write to file')
            tweets_df.to_csv(outfile)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file1 = luigi.Parameter(default='features.csv')
    output_file2 = luigi.Parameter(default='tweets_cities.csv')

    def requires(self):
        print('TrainingData requires')
        return CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        print('TrainingData output')
        return {'output1': luigi.LocalTarget(self.output_file1),
                'output2': luigi.LocalTarget(self.output_file2)}

    def run(self):
        print('TrainingData run')
        cities_raw = pd.read_csv(self.cities_file)
        cities_df = cities_raw[['name','latitude','longitude']]
        tweets_df = pd.read_csv(self.input().open('r'), encoding = 'unicode_escape', index_col=0)
        # encoding classes
        class_dict = {'negative':0,'neutral':1,'positive':2}
        tweets_df = tweets_df.replace({'airline_sentiment':class_dict})
        # separating lat and long and turning them to float
        def get_lat_long(string):
            lat = float(string.strip('[]').split(',')[0])
            long = float(string.strip('[]').split(',')[1])
            return lat, long
        tweets_df['lat'] = tweets_df['tweet_coord'].apply(lambda x: get_lat_long(x)[0])
        tweets_df['long'] = tweets_df['tweet_coord'].apply(lambda x: get_lat_long(x)[1])
        # turn lat/lon to Cartesian coordinates
        def get_3d_coords(lon, lat):
            x = math.cos(lat)*math.cos(lon) # radius is ignored because we only want a rank
            y = math.cos(lat)*math.sin(lon)
            z = math.sin(lat)
            return x, y, z
        # converting degrees to radians
        tweets_df['lat_rad'] = tweets_df['lat'].apply(math.radians)
        tweets_df['lon_rad'] = tweets_df['long'].apply(math.radians)
        cities_df['lat_rad'] = cities_df['latitude'].apply(math.radians)
        cities_df['lon_rad'] = cities_df['longitude'].apply(math.radians)
        # getting Cartesian coordinates
        tweets_df['x'] = pd.DataFrame(tweets_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[0], axis=1))
        tweets_df['y'] = pd.DataFrame(tweets_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[1], axis=1))
        tweets_df['z'] = pd.DataFrame(tweets_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[2], axis=1))
        cities_df['x'] = pd.DataFrame(cities_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[0], axis=1))
        cities_df['y'] = pd.DataFrame(cities_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[1], axis=1))
        cities_df['z'] = pd.DataFrame(cities_df.apply(lambda x: get_3d_coords(x['lon_rad'], x['lat_rad'])[2], axis=1))
        # getting nearest city using KNN (k=1) (much faster than calculating Euclidian distances by formula)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(cities_df[['x','y','z']], cities_df['name'])
        tweets_df['city'] = knn.predict(tweets_df[['x','y','z']])
        # one-hot encoding
        lb = preprocessing.LabelBinarizer()
        lb.fit(tweets_df['city'])
        tweets_df_oh = pd.DataFrame(lb.transform(tweets_df['city']), columns=lb.classes_)
        # preparing features for training
        tweets_df_oh = tweets_df_oh.reset_index(drop=True)
        tweets_df = tweets_df.reset_index(drop=True)
        train_df = pd.concat([tweets_df_oh,tweets_df[['airline_sentiment']]], axis=1)
        # saving to files
        with self.output()['output1'].open('w') as outfile1, self.output()['output2'].open('w') as outfile2:
            print('TrainingData write to file')
            train_df.to_csv(outfile1)
            tweets_df.to_csv(outfile2)



class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        print('TrainModel requires')
        return TrainingDataTask(tweet_file=self.tweet_file)

    def output(self):
        print('TrainModel output')
        return luigi.LocalTarget(self.output_file)

    def run(self):
        print('TrainModel run')
        tweets_df = pd.read_csv(self.input()['output1'].open('r'), index_col=0)
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(tweets_df.iloc[:, :-1], tweets_df['airline_sentiment'], test_size=0.2, random_state=42)
        # grid search for random forest classifier
        params = {'n_estimators':  [10, 50, 100, 200],
                  'max_depth': [None, 5, 10],
                  'max_features': ['auto','log2'],
                  'class_weight': ['balanced', 'balanced_subsample']}
        rfc_gs = GridSearchCV(RandomForestClassifier(), params, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
        rfc_gs.fit(X_train, y_train)
        # best model
        rfc_best = rfc_gs.best_estimator_
        rfc_best.fit(X_train, y_train)
        # the metrics
        print(f'Training accuracy: {rfc_best.score(X_train, y_train)}')
        print(f'Test accuracy: {rfc_best.score(X_test, y_test)}')
        print(f'Test F1 Score: {f1_score(y_test, rfc_best.predict(X_test), average="macro")}')
        print("\n----------- Baseline Classification Report -----------\n")
        print(classification_report(y_test, rfc_best.predict(X_test)))
        # saving model
        with open(self.output().path, 'wb') as outfile:
            print('TrainModel write to file')
            joblib.dump(rfc_best, outfile)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def requires(self):
        print('Score requires')
        return [TrainModelTask(tweet_file=self.tweet_file),
                TrainingDataTask(tweet_file=self.tweet_file)]

    def output(self):
        print('Score output')
        return luigi.LocalTarget(self.output_file)

    def run(self):
        print('Score run')
        # loading model
        rfc_pkl = joblib.load(open(self.input()[0].path, 'rb'))
        # getting one-hot coded cities
        tweets_df_oh = pd.read_csv(self.input()[1]['output1'].open('r'), index_col=0)
        tweets_df_oh = tweets_df_oh.iloc[:, :-1]
        # getting city names
        tweets_df = pd.read_csv(self.input()[1]['output2'].open('r'), index_col=0)
        # scoring
        sentiment_score = pd.DataFrame(rfc_pkl.predict_proba(tweets_df_oh), columns=['negative','neutral','positive'])
        # rearranging and sorting
        sentiment_score['city'] = tweets_df['city'].values
        sentiment_score = sentiment_score[['city','negative','neutral','positive']]
        city_rank = sentiment_score.sort_values('positive', ascending=False).drop_duplicates()
        city_rank = city_rank.reset_index(drop=True)
        # saving to file
        with self.output().open('w') as outfile:
            print('Score write to file')
            city_rank.to_csv(outfile)


if __name__ == "__main__":
    luigi.run()
