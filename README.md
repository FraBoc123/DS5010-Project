## Project 

### Installation and dev setup
Assumption
1. Windows/mac/linux os
2. has anaconda installed
3. pycharm as editor

**Installation**
1. Create conda environment
`conda create --name ml python=3.9`
2. Activate environment
`conda activate ml`
3. Install all the packages necessary
`conda install --yes -c conda-forge --file requirements.txt`

## To run the program
1. Activate conda environment
`conda activate ml`
2. Go to the src folder
`cd src`
3. Run main file
`python main.py`

## Options and caveats
1. All the index values expected to be number
2. For prediction of a game model is expected to be trained and input should contain
   - `Day` -Mon, Tue, Wed...Sun
   - `Venue` - Home, Away, Neutral
   - `Poss` - 1,2,3,4,...100

# Program Functionalities
1. Run the scrapper
2. Create model for spcecific team
3. Predict the game for a team
4. Quit


# Running test cases
All the test cases are written with unittest framework in python
You can run test suite from pycharm or from command line with
`python -m unittest` this command.

