# Kaggle Hackathon

Today you will be experimenting with different machine learning models for the beginner Kaggle competition using the Titanic dataset. You will work with your team members to try to improve upon the results for the random forest classifier demonstrated in the tutorial for this competition. Let's get started

## Teams
Here are the teams:

* Erick, Riki
* Ryan, Sam
* Lynn, Abby
* Lulu, Katie
* Irene, Allison
* Kate, Lina
* Maya, Grace
* Christine, Luke
* Jasmine, Kasey
* Sarah, Olivia
* Georgette, Bella
* Richard, Stephen	
* Zach, Salamun
* Ethan, Ike
* Quoc, Ivan
* Alba, Amelia
* Aditya, Nolan
* Rey, John
* Chris, Jackson
* Esther, Xiaowei


## 1. Getting started
Select one person on your team to be the submitter. Then do the following together with that person being the "driver". You'll be submitting a solution together as one team to the competition.

1. Follow the directions in the slides from Tuesday to create an account.
2. Open [the Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial) in one window.
3. In another window, join the ["Titanic - Machine Learning from Disaster" beginner competition](https://www.kaggle.com/competitions/titanic).
4. Follow along with the tutorial to create a new notebook.
5. Enter the code from the tutorial into the notebook.
6. Follow all the directions in the notebook to make your first submission. **Note** In the tutorial it says "Click on the Data tab on the top of the screen. Then, click on the Submit button to submit your results." There is no Data tab! Instead you click on the **"Output"** tab.

An interesting note about the code: in the code I've shared with you, I used the pandas `factorize()` function to convert "male" and "female" to `0` and `1`. I learned from this notebook that you can do the same thing with the function `get_dummies()`. From what I understand, `get_dummies()` only works for binary labels (0, 1), while factorize can create more than 2 labels if necessary. There are other differences as well.

## 2. Creating a dev (development) set from the training data
You are only allowed to submit three submissions per day to Kaggle (or something like that). In order to avoid being locked out of submitting, you must experiment with the training data until you think you've got a good model, and only then will you test on the official test data and submit your predictions to Kaggle.  

You will take the provided training data and partition it into a **training** set and  **development** set. You will build models with that new training set (a subset of the original training data) and test it on the development set. Then, when you've got good accuracy on that development set, you will train on the whole training set, test on the official test set, and submit your predictions to Kaggle.

How do you do this? You can use the same syntax as we used to partition our data in the last homework and in the recent sample code. In the Kaggle notebook, find this code:

```
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

```
After that code block insert a new code block containing a few lines to partition `X` and `y` into `X_train`, `X_dev`, `y_train`, and `y_dev`, like so:

```
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)
```

In the subsequent code blocks, you will build classifers using `X_train` and `y_train`, predict outputs for `X_dev`, and evaluate your predictions against `y_dev`. 

## 3. Building and evaluating new models with your train and dev set
Okay, now you need to start building new models with the train set your created and evaluating those models with the dev set. Here are some suggestions:

1. Build each of the four classifiers we've learned so far using your new train partition: [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier). Evaluate them using your new dev partition. 

2. Change the parameters of those models to see if you can find improvements. For example, for `RandomForestClassifier`, you can increase or decrease the number of estimators. Not sure what all the parameters are? Check out the documentation by following the links to the classes above. It's okay if you don't totally understand what the parameters are doing, but try to pick ones that make sense.

3. Change the features you use to do preditions. The baseline model in the tutorial uses just `["Pclass", "Sex", "SibSp", "Parch"]`. Try using fewer features or more features.

## 4. Submit your predictions

1. Once you have a model and a set of features you love, go to the `Save Version` button and save all your work. 

2. Now go back to the code cell where you originally created `submission.csv`. Change the features to the ones that did the best on your dev set. Here's what that line looks like to start with.

```
features = ["Pclass", "Sex", "SibSp", "Parch"]
```

3. Then change the line of code that initializes the mode so that you initialize the model with the parameterization that worked best for you. Here's that line of code:

```
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
```

4. Now run all the code up to and including that code block. Then follow the instructions in the tutorial for submitting your `submission.csv` file to the Kaggle competition,

5. Within a few minutes you'll be able to see your accuracy and where you are on the leaderboard. Go to the [main Titanic competition page](https://www.kaggle.com/c/titanic/), then click on the `Leaderboard` tab.

## 5. Continue refining your model
