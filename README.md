# Babel
End Goal: iPhone app to translate a foreign language in real-time

Licensed under the [MIT License](https://opensource.org/licenses/MIT)

### Install Libaries

```
brew install armadillo
brew install libsndfile
brew install gnuplot --with-qt
```

### Build Code

```
g++ file.cpp -o file -std=c++11
```

### Add Flags to use Libraries

Armadillo

```
-O2 -larmadillo
```

Libsndfile

```
-lsndfile
```

### Run Code
```
./file
```

# Roadmap
Simple logistic regression with gradient descent ✓
   
![logistic regression loss](screenshots/logistic_regression.png)
   
Graph loss over time steps ✓
   
![graph loss](screenshots/myopia_loss.png)
   
Spectrogram ✓
   
440 Hz sound, FFT Length of 1024

![1024 spectrogram](screenshots/1024_raw.png)
   
440 Hz sound, FFT Length of 128

![128 spectrogram](screenshots/128_raw.png)
   
Simple 1 hidden layer neural network ✓

![neural net loss](screenshots/h1_neural_net.png)

Multiclass classifier ✓

![Multiclass classifier loss](screenshots/multiclass.png)

Simple image classifier

Image classifier for different size images using pyramid kernel

Curve comparison algorithm for all frequencies

Lexicon of french phrases to english phrases

Audio files of all french words from Mac say tool

Labels for audio files

Train algorithm for words

Reduce dimensions using PCA

Train algorithm for words

Develop memory for parsing phrases

Lexicon of french phrases to english phrases ✓

Audio files of all french phrases from Mac say tool

Labels for audio files

Train algorithm for phrases
   
Deploy app on iPhone to hook up w/ wearable mic and microphone

Wearables: Zungle Panther / earbud

## External Datasets to Use:
   http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes , body measurements vs. existence of diabetes ✓
   
   https://www.umass.edu/statdata/statdata/stat-logistic.html , myopia factors vs. existence of myopia ✓
   
   http://yann.lecun.com/exdb/mnist/ , image of number vs. number
   
   http://marsyasweb.appspot.com/download/data_sets/ , files vs. music genre
   
   http://www.manythings.org/bilingual/ , translation corpus for english vs. other other languages ✓
   
   https://archive.ics.uci.edu/ml/datasets/Iris, flower measurements vs. iris type ✓

## References:
   Andrew Ng paper on speech classification

   http://papers.nips.cc/paper/3674-unsupervised-feature-learning-for-audio-classification-using-convolutional-deep-belief-networks.pdf
   
   Montreal paper on music genre classification
   
   http://ismir2010.ismir.net/proceedings/ismir2010-58.pdf

   DARPA Case Study for Arabic translation
   
   https://requester.mturk.com/case_studies/cs/darpa
   
   Google paper on offline speech recognition
   
   http://arxiv.org/pdf/1603.03185.pdf
