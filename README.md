# chord_progression

A Keras-based LSTM network to generate jazz chord progressions. Based on [lstm_real_book](https://github.com/keunwoochoi/lstm_real_book).

###Usage
1. Run `$ python main_lstm.py` to train a network.
2. Run `$ python sample_lstm.py` to generate new chord progressions.

###Example output
`
Diversity: 1.0
|    Fmaj    Fmaj    Fmaj    Fmaj |    Gmin    Gmin   Cdom7   Cdom7 |    Fmaj    Fmaj    Fmaj    Fmaj |    Gmin    Gmin   Cdom7   Cdom7 | 
|    Fmaj    Fmaj   Cdom7   Cdom7 |    Fmaj    Fmaj    Fmaj    Fmaj |    Fmaj    Fmaj    Cmaj    Fmaj |    Gmin    Gmin   Cdom7   Cdom7 | 
|    Fmaj    Fmaj    Fmin    Fmin |    Amin    Amin   Ddom7   Ddom7 |    Gmin    Gmin    Gmin    Gmin |   Cdom7   Cdom7   Cdom7   Cdom7 | 
|    Fmaj    Fmaj    Fmaj    Fmaj |   Fmin6   Fmin6   Fmin6   Fmin6 |    Fmaj    Fmaj    Fmaj    Fmaj |    Fmaj    Fmaj    Fmaj    Fmaj
---
`

For more examples see [examples/gen_sequences.txt](https://raw.githubusercontent.com/hochthom/chord_progression/master/examples/gen_sequences.txt).

