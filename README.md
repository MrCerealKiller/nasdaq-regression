# NASDAQ Stock Analysis Tool

## Description
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

## Data Format
The data from the csv is from the NASDAQ archive
It has the standard form below (by column index):
0. Date
1. Closing Price
2. Volume Traded
3. Opening Price
4. High
5. Low

## Theory
The aim of this module is to define the current day's closing price as a
function of the last k days worth of data.

Thus the datastructure is:

```
Y_n = f([close_(n-1), volume_(n-1), open_(n-1), high_(n-1), low_(n-1)],
        [close_(n-2), volume_(n-2), open_(n-2), high_(n-2), low_(n-2)],
        [close_(n-3), volume_(n-3), open_(n-3), high_(n-3), low_(n-3)],
        [close_(n-4), volume_(n-4), open_(n-4), high_(n-4), low_(n-4)],
        ...
        [close_(n-k), volume_(n-k), open_(n-k), high_(n-k), low_(n-k)])
```

## Dependencies
- Tensorflow
- Keras
- Pandas
- Matplotlib (Optional | Visualization)
- Pickle (Optional | CLI)
- Argparse (Optional | CLI)

These can all be downloaded and managed through Pip.

## Usage
The module can be imported and used in a larger application, but there is
also a simple command line interface.

The basic usage for is:
`./stock_analyzer.py <TRAINING_DATA.csv> <TESTING_DATA.csv>`

For more advanced usage (importing and exporting learned states) please see
the help menu.

