#!/usr/bin/env python

'''
###############################################################################

    STOCK ANALYSIS TOOL

###############################################################################

By: Jeremy Mallette (MrCerealKiller)
Date Last Updated: 29/08/2018

-------------------------------------------------------------------------------

This is a tool for analyzing NASDAQ historical financial data
It is not intended for any serious use...

Currently it is using the Keras API on top of Google Tensorflow
By looking back at 'n' days historical data, it attempts to predict
that day's closing price.

It provides a class that can be imported and used in other scripts,
but can also be run itself as a limited CLI

It could be easily expanded (and I hope to someday expand upon it myself),
by removing Keras and getting more opportunity to fine-tune all the knobs

Also adding more interesting data, such as '% of global max' or some sort of
public opinion metric could be nice, though the latter would be extremely
hard to get

-------------------------------------------------------------------------------

As noted below:

    The data from the csv is from the NASDAQ archive
    It has the standard form below (by column index):
        0: Date
        1: Close
        2: Volume
        3: Open
        4: High
        5: Low

Also as noted below:

    The aim is to define the current day's closing price as a function
    of the last k days worth of data.

    Thus the datastructure is:

    Y_n = f([close_(n-1), volume_(n-1), open_(n-1), high_(n-1), low_(n-1)],
            [close_(n-2), volume_(n-2), open_(n-2), high_(n-2), low_(n-2)],
            [close_(n-3), volume_(n-3), open_(n-3), high_(n-3), low_(n-3)],
            [close_(n-4), volume_(n-4), open_(n-4), high_(n-4), low_(n-4)],
            ...
            [close_(n-k), volume_(n-k), open_(n-k), high_(n-k), low_(n-k)])
'''

import sys
from time import gmtime, strftime
import numpy as np

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse


# Global Process Flags
VERBOSE = True
EXPORT = False


class fm:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


# =============================================================================
class StockAnalyzer(object):
    def __init__(self,
                 train_filename='stock_data/test.train.csv',
                 test_filename='stock_data/test.test.csv',
                 states_dir='states/',
                 models_dir='models/',
                 days_back=5,
                 sample_size=1000,
                 epochs=500,
                 es_patience=100,
                 validation_split=0.2,
                 rms_optimizer=0.001,
                 tf_verbosity=0,
                 display_rate=1):

        self.import_state = False    # Use an exported dataset or not
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.state_filename = ''
        self.states_dir = states_dir
        self.models_dir = models_dir

        self.days_back = days_back
        self.sample_size = sample_size
        self.entry_shape = (self.days_back, 5)

        self.epochs = epochs
        self.es_patience = es_patience
        self.validation_split = validation_split
        self.rms_optimizer = rms_optimizer
        self.tf_verbosity = tf_verbosity
        self.display_rate = display_rate

        if VERBOSE is True:
            print(fm.OKBLUE + '----------------------------------------')
            print(fm.BOLD + 'Initialized with these parameters:' + fm.ENDC)

            print('Training File:\t\t{}'.format(self.train_filename))
            print('Testing File:\t\t{}'.format(self.test_filename))
            print('Days to Analyze:\t{}'.format(self.days_back))
            print('Sample Size:\t\t{}'.format(self.sample_size))
            print('')
            print('Epochs:\t\t\t{}'.format(self.epochs))
            print('Patience:\t\t{}'.format(self.es_patience))
            print('Validation Split:\t{}'.format(self.validation_split))
            print('RMS Opt. Coeff.:\t{}'.format(self.rms_optimizer))

            print(fm.OKBLUE + '----------------------------------------\n' +
                  fm.ENDC)

# =============================================================================
    def get_train_filename(self):
        return self.train_filename

    def set_train_filename(self, filename):
        self.train_filename = filename

    def get_test_filename(self):
        return self.test_filename

    def set_test_filename(self, filename):
        self.test_filename = filename

    def get_days_back(self):
        return self.days_back

    def set_days_back(self, days):
        self.days_back = days

    def get_sample_size(self):
        return self.sample_size

    def set_sample_size(self, size):
        self.sample_size = size

    def get_epochs(self):
        return self.epochs

    def set_epochs(self, epochs):
        self.epochs = epochs

    def get_es_patience(self):
        return self.es_patience

    def set_es_patience(self, es_patience):
        self.es_patience = es_patience

    def get_validation_split(self):
        return self.validation_split

    def set_validation_split(self, val):
        self.validation_split = val

    def get_rms_optimizer(self):
        return self.rms_optimizer

    def set_rms_optimizer(self, opt):
        self.rms_optimizer = opt

    def get_tf_verbosity(self):
        return self.tf_verbosity

    def set_tf_verbosity(self, verbosity):
        self.tf_verbosity = verbosity

    def get_display_rate(self):
        return self.display_rate

    def set_display_rate(self, rate):
        self.info_print_rate = rate

# ###########################
# ##                       ##
# ##  Data Set Prepartion  ##
# ##                       ##
# ###########################

# =============================================================================
    def override_import_state(self, filename):
        if VERBOSE is True:
            print(fm.WARNING + 'Overriding training data file with object ' +
                  'from: \'{}\''.format(filename) + fm.ENDC)

        self.import_state = True
        self.state_filename = filename

# =============================================================================
    def read_train_data_from_csv(self):
        csv = pd.read_csv(self.train_filename, header=None, index_col=None)

        '''
        The data from the csv is from the NASDAQ archive
        It has the standard form below (by column index):
            0: Date
            1: Close
            2: Volume
            3: Open
            4: High
            5: Low
        '''

        if VERBOSE is True:
            print('Loading training data from ' +
                  '\'{}\''.format(self.train_filename))

        if len(csv.columns) > 6:
            print(fm.WARNING)
            print('Please ensure that you have the correct data structure')
            print('Found {} Columns - 6 Expected'.format(len(csv.columns)))
            print('Continuing regardless...\n' + fm.ENDC)
        elif len(csv.columns) < 6:
            raise Exception('Not enough data columns - Expected 6 -' +
                            'Found {}'.format(len(csv.columns)))

        self.train_raw = csv[csv.columns[1:6]].values
        self.available_train_data = len(self.train_raw) - self.days_back

        if len(self.train_raw) <= self.days_back:
            raise Exception('The dataset is not large enough to look back.')

        if self.sample_size > self.available_train_data:
            print(fm.WARNING)
            print('Given sample size is larger than dataset.\n' +
                  'The sample size will be truncated to fit the data.\n' +
                  fm.ENDC)
            self.sample_size = self.available_train_data

        self.train_data_shape = ((self.available_train_data, ) +
                                 self.entry_shape)

# =============================================================================
    def read_test_data_from_csv(self):
        csv = pd.read_csv(self.test_filename, header=None, index_col=None)

        '''
        The data from the csv is from the NASDAQ archive
        It has the standard form below (by column index):
            0: Date
            1: Close
            2: Volume
            3: Open
            4: High
            5: Low
        '''

        if VERBOSE is True:
            print('Loading testing data from ' +
                  '\'{}\''.format(self.test_filename))

        if len(csv.columns) > 6:
            print(fm.WARNING)
            print('Please ensure that you have the correct data structure')
            print('Found {} Columns - 6 Expected'.format(len(csv.columns)))
            print('Continuing regardless...\n' + fm.ENDC)
        elif len(csv.columns) < 6:
            raise Exception('Not enough data columns - Expected 6 -' +
                            'Found {}'.format(len(csv.columns)))

        self.test_raw = csv[csv.columns[1:6]].values
        self.available_test_data = len(self.test_raw) - self.days_back
        self.test_data_shape = (self.available_test_data, ) + self.entry_shape

        if len(self.test_raw) <= self.days_back:
            raise Exception('The dataset is not large enough to look back.')


# =============================================================================
    def calculate_stats(self):
        if VERBOSE is True:
            print('Calculating the columnal means and standard deviations' +
                  'from training data')

        self.means = np.mean(self.train_raw, axis=0)
        self.stds = np.std(self.train_raw, axis=0)

        if VERBOSE is True:
            print(fm.OKGREEN + '----------------------------------------')
            print('Total Available Data Entries: ' +
                  '{}'.format(self.available_train_data) + fm.ENDC)

            print('\nMeans:')
            print('\tClose: {}'.format(self.means[0]))
            print('\tVolume: {}'.format(self.means[1]))
            print('\tOpen: {}'.format(self.means[2]))
            print('\tHigh: {}'.format(self.means[3]))
            print('\tLow: {}'.format(self.means[4]))
            print('\nStandard Deviations:')
            print('\tClose: {}'.format(self.stds[0]))
            print('\tVolume: {}'.format(self.stds[1]))
            print('\tOpen: {}'.format(self.stds[2]))
            print('\tHigh: {}'.format(self.stds[3]))
            print('\tLow: {}'.format(self.stds[4]))
            print(fm.OKGREEN + '\n----------------------------------------' +
                  fm.ENDC)

# =============================================================================
    def prepare_data_set(self):
        '''
        The aim is to define the current day's closing price as a function
        of the last k days worth of data.

        Thus the datastructure is:

        Y_n = f([close_(n-1), volume_(n-1), open_(n-1), high_(n-1), low_(n-1)],
                [close_(n-2), volume_(n-2), open_(n-2), high_(n-2), low_(n-2)],
                [close_(n-3), volume_(n-3), open_(n-3), high_(n-3), low_(n-3)],
                [close_(n-4), volume_(n-4), open_(n-4), high_(n-4), low_(n-4)],
                ...
                [close_(n-k), volume_(n-k), open_(n-k), high_(n-k), low_(n-k)])
        '''

        if VERBOSE is True:
            print('Preparing training data and choosing samples randomly')

        if not (self.train_raw.any() or
                self.test_raw.any()):
            raise Exception('Required data is not available.\n' +
                            'Please load the training and testing sets from ' +
                            'a file and run calculate_stats().')

        self.train_data = np.zeros(shape=self.train_data_shape)
        self.train_labels = np.zeros(shape=(self.sample_size))

        self.test_data = np.zeros(shape=self.test_data_shape)
        self.test_labels = np.zeros(shape=(self.available_test_data))

        # Training Data

        start_idx = self.days_back
        end_idx = len(self.train_raw)

        sample_indexes = np.random.choice(range(start_idx, end_idx),
                                          size=self.sample_size,
                                          replace=False)

        count = 0
        for i in sample_indexes:
            # Look at least one day back to get some data
            entry = [self.train_raw[i - 1]]
            # Add preceding days as per look back criteron
            for j in range(1, self.days_back):
                entry = np.append(entry, [self.train_raw[(i - 1) - j]], axis=0)

            self.train_data[count] = entry
            self.train_labels[count] = self.train_raw[i, 0]
            count += 1

        if VERBOSE is True:
            print('Preparing testing data')

        # Testing Data

        end_idx = len(self.test_raw)

        count = 0
        for i in range(start_idx, end_idx):
            # Look at least one day back to get some data
            entry = [self.test_raw[i - 1]]
            # Add preceding days as per look back criteron
            for j in range(1, self.days_back):
                entry = np.append(entry, [self.test_raw[(i - 1) - j]], axis=0)

            self.test_data[count] = entry
            self.test_labels[count] = self.test_raw[i, 0]
            count += 1

# =============================================================================
    def normalize_data_set(self):
        if not (self.means.any() or
                self.stds.any()):
            raise Exception('Required data is not available.\n' +
                            'Please load the training and testing sets from ' +
                            'a file and run calculate_stats().')

        if VERBOSE is True:
            print('Performing Z-Score Normalization on training data')

        for i in range(len(self.means)):
            self.train_data[:, :, i] = ((self.train_data[:, :, i] -
                                        self.means[i]) / self.stds[i])

        if VERBOSE is True:
            print('Performing Z-Score Normalization on testing data')

        for i in range(len(self.means)):
            self.test_data[:, :, i] = ((self.test_data[:, :, i] -
                                       self.means[i]) / self.stds[i])

# =============================================================================
    def export_state(self):
        name = ('state_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.pkl')
        filepath = self.states_dir + name

        if VERBOSE is True:
            print('Exporting state to \'{}\''.format(filepath))\

        state = {}
        state['days_back'] = self.days_back
        state['sample_size'] = self.sample_size
        state['entry_shape'] = self.entry_shape
        state['means'] = self.means
        state['stds'] = self.stds

        state['available_train_data'] = self.available_train_data
        state['train_data_shape'] = self.train_data_shape
        state['train_data'] = self.train_data

        state['available_test_data'] = self.available_test_data
        state['test_data_shape'] = self.test_data_shape
        state['test_data'] = self.test_data

        try:
            with open(filepath, 'wb') as file:
                pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise e

# =============================================================================
    def import_state(self):
        if VERBOSE is True:
            print('Importing state from {}'.format(self.state_filename))

        try:
            with open(self.filename, 'rb') as file:
                state = pickle.load(file)

                if state.any():
                    self.days_back = state['days_back']
                    self.sample_size = state['sample_size']
                    self.entry_shape = state['entry_shape']
                    self.means = state['means']
                    self.stds = state['stds']

                    self.available_train_data = state['available_train_data']
                    self.train_data_shape = state['train_data_shape']
                    self.train_data = state['train_data']

                    self.available_test_data = state['available_test_data']
                    self.test_data_shape = state['test_data_shape']
                    self.test_data = state['test_data']

        except Exception as e:
            raise e

# =============================================================================
    def initialize_data(self):
        if self.import_state:
            self.override_import_state()
        else:
            # Attempt to read the indicated CSV file and perform some checks
            # on the data against some of the given parameters
            try:
                self.read_train_data_from_csv()
            except Exception as e:
                print(fm.FAIL + '\nError while loading training set:\n' +
                      '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                      fm.ENDC)
                sys.exit(1)
                return

            try:
                self.read_test_data_from_csv()
            except Exception as e:
                print(fm.FAIL + '\nError while loading testing set:\n' +
                      '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                      fm.ENDC)
                sys.exit(1)
                return

            # Calculate the mean and columnal standard deviation
            try:
                self.calculate_stats()
            except Exception as e:
                print(fm.FAIL + '\nError while calculating raw stats:\n' +
                      '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                      fm.ENDC)
                sys.exit(1)
                return

            # Prepare the samples to be used in the dataset
            try:
                self.prepare_data_set()
            except Exception as e:
                print(fm.FAIL + '\nError while preparing the dataset:\n' +
                      '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                      fm.ENDC)
                sys.exit(1)
                return

            # Perform Z-Score Normalization on the dataset
            try:
                self.normalize_data_set()
            except Exception as e:
                print(fm.FAIL + '\nError while normalizing the dataset:\n' +
                      '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                      fm.ENDC)
                sys.exit(1)
                return

            if EXPORT is True:
                self.export_state()

# ########################
# ##                    ##
# ##  Machine Learning  ##
# ##                    ##
# ########################

# =============================================================================
    def build_model(self):
        if VERBOSE is True:
            print('Building the intitial model')

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=self.entry_shape),
            keras.layers.Dense(64, activation=tf.nn.relu,
                               input_shape=(self.train_data.shape[1],)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        model.compile(loss='mse',
                      optimizer=tf.train.RMSPropOptimizer(self.rms_optimizer),
                      metrics=['mae'])

        self.model = model

# =============================================================================
    def train_model(self):
        if VERBOSE is True:
            print('Training...\n')

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=self.es_patience)

        progress_handler = StockAnalyzer.ProgressHandler(self.display_rate,
                                                         self.epochs)

        self.history = self.model.fit(self.train_data,
                                      self.train_labels,
                                      epochs=self.epochs,
                                      validation_split=self.validation_split,
                                      verbose=self.tf_verbosity,
                                      callbacks=[early_stop,
                                                 progress_handler])

        if VERBOSE is True:
            print('\n')

# =============================================================================
    def test_model(self):
        if VERBOSE is True:
            print('Testing...')

        [self.loss, self.mae] = self.model.evaluate(self.test_data,
                                                    self.test_labels,
                                                    verbose=self.tf_verbosity)

        self.test_results = self.model.predict(self.test_data).flatten()
        self.test_error = self.test_results - self.test_labels

        print(fm.OKGREEN + '----------------------------------------' +
              fm.BOLD)
        print('  MAE:\t${:.2f}'.format(self.mae))
        print('  Loss:\t{:.2f}'.format(self.loss) + fm.ENDC)
        print(fm.OKGREEN + '----------------------------------------' +
              fm.ENDC)

# =============================================================================
    def save_model(self):
        name = ('model_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.h5py')
        filepath = self.models_dir + name

        keras.models.save_model(self.model,
                                filepath,
                                overwrite=True,
                                include_optimizer=False)

        if VERBOSE is True:
            print('Saved model to \'{}\''.format(filepath))

# =============================================================================
    def execute(self):
        # Build the Model
        try:
            self.build_model()
        except Exception as e:
            print(fm.FAIL + '\nError while building the model:\n' +
                  '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                  fm.ENDC)
            sys.exit(1)
            return
        # Train the Model
        try:
            self.train_model()
        except Exception as e:
            print(fm.FAIL + '\nError while training the model:\n' +
                  '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                  fm.ENDC)
            sys.exit(1)
            return

        # Test the Model
        try:
            self.test_model()
        except Exception as e:
            print(fm.FAIL + '\nError while testing the model:\n' +
                  '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                  fm.ENDC)
            sys.exit(1)
            return

# #####################
# ##                 ##
# ##  Visualization  ##
# ##                 ##
# #####################

# =============================================================================
    class ProgressHandler(keras.callbacks.Callback):
        def __init__(self, display_rate, goal):
            self.display_rate = display_rate
            self.goal = goal
            self.length = 10

            super(self.__class__, self).__init__()

        def on_epoch_end(self, epoch, logs):
            if VERBOSE is not True:
                return

            prog = ((epoch * 1.0) / (self.goal * 1.0)) * 100.0
            full = int(round((prog * 1.0) / (self.length * 1.0)))

            bar = '\r['
            for _ in range(full):
                bar += '#'
            for _ in range(self.length - full):
                bar += '-'
            bar += '] : {}% (Epoch {}/{})'.format(prog, epoch, self.goal)

            sys.stdout.write(bar)
            sys.stdout.flush()

# =============================================================================
    def plot_evolution(self):
        if VERBOSE is True:
            print('Preparing an evolution plot')

        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('MAE [$]')
        plt.plot(self.history.epoch,
                 np.array(self.history.history['mean_absolute_error']),
                 label='Train Loss')
        plt.plot(self.history.epoch,
                 np.array(self.history.history['val_mean_absolute_error']),
                 label='Val Loss')
        plt.legend()
        plt.show()

# =============================================================================
    def plot_predications(self):
        if VERBOSE is True:
            print('Preparing a predication plot')

        plt.figure(2)
        plt.xlabel('True Values [$]')
        plt.ylabel('Predictions [$]')
        plt.scatter(self.test_labels, self.test_results)
        plt.axis('equal')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot([-100, 100], [-100, 100])
        plt.show()

        if VERBOSE is True:
            print('Preparing a history of error')

        plt.figure(3)
        plt.xlabel("Prediction Error [$]")
        plt.ylabel("Count")
        plt.hist(self.test_error, bins=50)
        plt.show()

# =============================================================================
    def visualize_results(self):
        # Plot the evolution
        try:
            self.plot_evolution()
        except Exception as e:
            print(fm.FAIL + '\nError while plotting the evolution:\n' +
                  '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                  fm.ENDC)
            sys.exit(1)
            return
        # Plot prediction information
        try:
            self.plot_predications()
        except Exception as e:
            print(fm.FAIL + '\nError while plotting the predictions:\n' +
                  '{}\n'.format(e.message) + fm.BOLD + 'Aborting...' +
                  fm.ENDC)
            sys.exit(1)
            return

# #############################
# ##                         ##
# ##  Commandline Interface  ##
# ##                         ##
# #############################


# =============================================================================
if __name__ == '__main__':
    # Setup argument parser ---------------------------------------------------
    parser = argparse.ArgumentParser("CLI for a stock price analyzer built " +
                                     "on the Google Tensorflow framework")

    parser.add_argument('train_file', metavar='TRAIN_FILE',
                        help='path to training data')
    parser.add_argument('test_file', metavar='TEST_FILE',
                        help='path to testing data')
    parser.add_argument('-i', '--import_file',
                        help='Precedes a filename for a saved state')
    parser.add_argument('-e', '--export',
                        help='Export samples to a pkl file for later use',
                        action='store_true')
    parser.add_argument('-q', '--quiet',
                        help='Decrease output verbosity',
                        action='store_true')

    args = parser.parse_args()

    # Handle optional arguments
    if args.quiet:
        VERBOSE = False

    if args.export:
        EXPORT = True

    # Print header
    if VERBOSE is True:
        print(fm.HEADER + fm.BOLD + '\n' +
              '  #########################\n' +
              '  ## Stock Analysis Tool ##\n' +
              '  #########################\n' + fm.ENDC)

    # Create StockAnalyzer object and handle input files
    if args.train_file and args.test_file:
        sa = StockAnalyzer(train_filename=args.train_file,
                           test_filename=args.test_file)
    else:
        sa = StockAnalyzer()

    if args.import_file:
        sa.override_import_state(args.import_file)

    # Initialize and run
    sa.initialize_data()
    sa.execute()
    sa.visualize_results()

    # Save the Keras model
    save = raw_input('Would you like to save this model? (y/N) ')
    if save == 'y':
        sa.save_model()

    if VERBOSE is True:
        print(fm.OKGREEN + fm.BOLD + '\nCompleted successfully\n')
