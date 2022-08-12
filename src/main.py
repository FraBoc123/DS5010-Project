from sklearn.model_selection import train_test_split
from model_evaluator.evaluate_model import ModelEvaluator
from scraper.scraper import Scrapper
from models.ml_logistic_regression import MLModelLogisticRegression
from models.ml_svc import MLSupportVectorClassifier
from models.ml_xgboost import MLXGBoostClassifier

import pandas as pd

def train_team_model(df):
    model = None
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Result']), df['Result'],
                                                        test_size=0.2,
                                                        random_state=2,
                                                        stratify=df['Result'])
    print('training model')
    choice = -1
    while not(0 < choice < 4):
        try:
            print('Select the model type ')
            print('1. Logistic Regression')
            print('2. Support Vector Machine')
            print('3. XGBoost')
            choice = int(input('Your option: '))
        except:
            print('Please select a valid option')
    if choice == 1:
        model = MLModelLogisticRegression()
    elif choice == 2:
        model = MLSupportVectorClassifier()
    else:
        model = MLXGBoostClassifier()

    model.train(X_train, y_train)
    print('Predicting on training data')

    # Check the train error
    y_train_pred = model.predict(X_train)
    eval_train, train_acc, train_f1 = ModelEvaluator.metrics_score(y_train, y_train_pred)
    print("Confusion matrix for Training:")
    print(eval_train)
    print(f'Training accuracy: {train_acc}, along with F1-Score score: {train_f1}\n\n')

    # Check the test error
    y_test_pred = model.predict(X_test)
    eval_test, test_acc, test_f1 = ModelEvaluator.metrics_score(y_test, y_test_pred)
    print("Confusion matrix for Testing: ")
    print(eval_test)
    print(f'Testing accuracy: {test_acc}, along with F1-Score score: {train_f1}\n\n')
    return model


def main():

    # Read data and drop redundant column
    data = pd.read_excel('data/Teams.xlsx')

    print('-----------------Welcome----------------')
    option = 'o'
    models = {}
    available_teams = None
    while(option !='4'):
        print('SELECT YOUR OPTION')
        print('1. Scrap the data')
        print('2. Train and test models for a particular team')
        print('3. Perform next game prediction')
        print('4. Quit the program')
        option = input('Select Your Option')

        if option == '1':
            print('Starting the scrapping process!!')
            scrapper = Scrapper("https://fbref.com/en/comps/9/11160/2021-2022-Premier-League-Stats",
                                "data/Teams.xlsx")
            scrapper.get_data()
            data = pd.read_excel('data/Teams.xlsx')
            print('Scrapping completed!!')
        elif option == '2':
            print('Please select team for which you would like to create the models')
            available_teams = data['Team'].unique().tolist()
            choice = -1
            while choice < 0:
                try:
                    for ind, team in enumerate(available_teams):
                        print(f'{ind + 1 }. {team}')
                    choice = int(input('Choose team index'))
                except:
                    print('Please choose from indices that are shown')
            if 0 < choice <= len(available_teams):
                team = available_teams[choice - 1]
                print(f'You selected {team}')
                df = data[data['Team'] == team]
                models[team] = train_team_model(df[['Day', 'Venue', 'Result', 'Poss']])
            else:
                print('Please select a valid team index!!')
        elif option == '3':
            if available_teams:
                print("Choose team for next game prediction (If you don't see your team below please train the models)")
                model_teams = list(models.keys())
                for ind, key in enumerate(model_teams):
                    print(f"{ind}, {key}")
                choice = int(input('Select team index!!'))
                model = models[model_teams[choice]]
                if model:
                    done = False
                    while not done:
                        try:
                            print('Lets try to predict the next game result!!')
                            features = input('Please insert expected Day, Venue, Possession for prediction')
                            day, venue, possession = features.strip('(').strip(')').split(', ')
                            day = day.lower().capitalize()
                            venue = venue.lower().capitalize()
                            poss = int(possession)
                            test = pd.DataFrame.from_records([[day, venue, poss]], columns=['Day', 'Venue', 'Poss'])
                            pred = model.predict(test)
                            print(f'Model Predicted it will be a {pred[0]}!!')
                            done = True
                        except:
                            print('please enter valid input, have a read at README.md')
            else:
                print('Train model for the team first!!')

        elif option == '4':
            print('GOODBYE!!')
        else:
            print('No valid option was selected please try again')


if __name__ == '__main__':
    main()

#when you train multiple models, on multiple teams, and you choose which team to make a prediction on, it doesnt keep track of the models that were used