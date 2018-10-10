# Zero-Shot-Learning
A pyhton ZSL system which makes it easy to run Zero-Shot Learning on new datasets, by giving it features and attributes. Based on “An embarrassingly simple approach to zero-shot learning”, presented at ICML 2015. 

## Dependencies
The dependencies can be found in the requirements.txt file. These can be installed by doing `pip install -r path/to/requirements.txt`.

## Usage
The system is designed to be easy to setup and run. In the folowing sections we will state how the system can be run.

### Files
The system uses three files:
 - `Xtrain`: this includes all the features for every instance the training set.
 - `Xtest`: this includes all the features for every instance in the testing set.
 - `Smatrix`: this includes all the attributes of all the classes.

 All the files should be kept in the same directory while running the system. The files should be named _exactly_ like stated above.

 ### Format
 The X files should contain lines for every instance: Each line starts with the class number and is followed by an undefined number of features. The features must range from 0 to 1. The class numbers should start at 0 and go up to the number of classes that are available in the complete dataset minus one (so training set and test set).

 The S file should contain lines for every class: every line starts with the classname or classnumber (this is not used in the code, so you can decide yourself) and is followed by the attribute values for said class.

### Running
First a `ZSL` object has to be created, which needs the path to the dataset.

The parameters of the values have to be set in the python code. This is done by creating a `ZSL` object and performing `set_parameters()`. This takes three lists of parameter values. Note that the value `0` in the `kernel_sigmas` denotes the use of the linear kernel.

After the creation of a `ZSL` object, when the parameters are all set, the system is ready. To run the system `run()` can be called, this returns a result.
The result can then be used to output files or show the performance in the terminal.

For an example, look at `Run.py`.

### Preprocessing Scripts
An S matrix can automatically be generated in two formats; Default and Split. The default version gives you an S Matrix with X + 1 attributes per word. The Split version gives 4 * X attributes + 1 per word.

These scripts can be found as `createS-4Split.py` and `createS.py`.

They both take 3 arguments:
 - An alphabet file (this holds all the attributes per letter letter in the script)
 - A file containing all the words you want to process. Each line needs to start with a number, followed by a space and then followed by the word (Example below)
 - The output file

Two alphabet files are already included in the directory. These can be used for latin characters. Other alphabet files for other scripts (like Arabic) could be created in a similar manner.

So, in conclusion, the scripts can be called like this: `python3 createS.py alphabet.csv words.txt Smatrix`.

##### Word file example:
 ```
 1 testing
 2 other
 3 zero
 ```

## Cite
If you make use of this software or want to refer to it, please cite the following paper: __"Zero-Shot Learning Based Approach For Medieval Word Recognition Using Deep-Learned Features", published in ICFHR2018.__
