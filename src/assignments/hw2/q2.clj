(ns assignments.hw2.q2
  (:require
   [assignments.hw2.utils :refer :all]
   ;; [scicloj.sklearn-clj]
   [scicloj.hanamicloth.v1.api :as haclo]
   [libpython-clj2.python :refer [py. py.-]]
   [libpython-clj2.require :refer [require-python]]
   [my-py-clj.config :refer :all]
   [tablecloth.api :as tc]
   [scicloj.kindly.v4.kind :as kind]))

(question "Question 2")
(sub-question "Q2: Cross Validation for Ridge Regression (40 points)")

(md 
 "For this question, you will need to perform Ridge ($L^2$) regression with K-fold cross validation via Scki-Learn. The data set is the famous Boston Housing data, which is a benchmark data for regression. The data contains 14 attributes and the `medv`, median value of owner-occupied homes in $1000s, will be your target to predict. The data set can be downloaded from canvas. You can also find some helpful information at [Kaggle](https://www.kaggle.com/code/henriqueyamahata/boston-housing-with-linear-regression/notebook). 
  
  In lecture, we learned training, test, and validation. However, in the real-world, we cannot always afford to implement it due to insufficient data. An alternative solution is K-fold cross validation which uses a part of the available data to fit the model, and a different part to test it. K-fold CV procedure splits the data into K equal-sized parts.")


(sub-sub "1) Load the train data and test data")

(require-python '[numpy :as np])
(require-python '[pandas :as pd])
(require-python '[sklearn.model_selection :as model-selection])
(require-python '[sklearn.linear_model :as linear-model])
(require-python '[sklearn.metrics :as metrics])

(md "These lines import necessary Python libraries for data manipulation, model selection, and evaluation. I wish I hide the `:ok` code-outputs, alas, I haven't found that option yet.")

(def train-data (pd/read_csv "data/A1Q2_Train_Data.csv"))
(def test-data (pd/read_csv "data/A1Q2_Test_Data.csv"))

(md "Here we load the training and test datasets from CSV files using pandas.")

(pd/DataFrame train-data)
(pd/DataFrame test-data)

(md "Inspect the loaded datasets as pandas DataFrames.")

(def X-train (py. train-data drop "medv" :axis 1))
(def y-train (py. train-data "get" "medv"))

(md "We separate the features (X-train) and target variable (y-train) from the training data.")

(sub-sub "2) Perform Ridge regression on the train data")

(def ridge-model (linear-model/Ridge))
(def param-grid {"alpha" [150 160 165 170 175 180 185 190 195 200]})
(def cv-ridge (model-selection/GridSearchCV ridge-model param-grid :cv 5))

(md "Here we set up the Ridge regression model and define a parameter grid for alpha values. We then create a GridSearchCV object for 5-fold cross-validation.")

(py. cv-ridge fit X-train y-train)

(md "This line fits the Ridge regression model using GridSearchCV on the training data.")

(answer (str "Best parameters:" (py.- cv-ridge best_params_)))
(answer (str "Best CV score:" (py.- cv-ridge best_score_)))

(md 
 "### K-fold Cross Validation Procedure:

1) The training data is divided into 5 equal parts (folds).
2) For each fold:

   a) That fold is treated as a validation set.
   b) The model is trained on the remaining 4 folds.
   c) The model's performance is evaluated on the validation fold.

3) This process is repeated 5 times, with each fold serving as the validation set once.
4) The average performance across all 5 validations is used as the cross-validation score.
5) This entire procedure is repeated for each hyperparameter combination (different alpha values).
6) The hyperparameters that yield the best average performance are selected.

 This method provides a robust estimate of the model's performance and helps in selecting the best hyperparameters, reducing the risk of overfitting.")

(md "Below, we print the best model's intercept and coefficients, and construct a human-readable equation for the Ridge regression model.")

(let [best-model (py.- cv-ridge best_estimator_)
      intercept (py.- best-model intercept_)
      coefficients (py.- best-model coef_)
      feature-names (py.- X-train columns)]
  (answer
   (str "Ridge Regression Equation:\n"
        "$medv = "
        (format "%.4f" intercept)
        (apply str
               (map (fn [name coef]
                      (format " %s %.4f * %s"
                              (if (pos? coef) "+" "-")
                              (Math/abs coef)
                              name))
                    feature-names
                    coefficients)) "$")))

(sub-sub "3) Justify the choice of K")

(md "We chose $K=5$ for cross-validation as it provides a good balance between bias and variance. It's a common choice that works well for most datasets, offering reliable performance estimates without excessive computational cost.")

(sub-sub "4) Test the model on the test data")

(def best-ridge-model (py.- cv-ridge best_estimator_))

(md "Here we extract the best Ridge regression model from the GridSearchCV results.")

^:kindly/hide-code
(comment
  (def X-test (py. test-data drop "medv" :axis 1))
  (def y-test (py. test-data "get" "medv"))

  (def y-pred (py. best-ridge-model predict X-test))
  (def mse (metrics/mean_squared_error y-test y-pred))
  (def r2 (metrics/r2_score y-test y-pred))

  (println "Mean Squared Error:" mse)
  (println "R-squared Score:" r2)

  (md "This commented-out code block shows how to separate features and target in the test data, make predictions, and calculate Mean Squared Error and R-squared score. It's likely commented out to avoid accidentally revealing test set performance during development."))



(def y-pred (py. best-ridge-model predict test-data))

(md "`y-pred` is the predictions on the entire test dataset using the best Ridge regression model.")

(sub-sub "5) Analyze the importance of each feature and justify your results in the report")

(def feature-importance (py.- best-ridge-model coef_))
(def feature-names (py.- X-train columns))

(md "These lines extract the feature coefficients (which represent feature importance in linear models) and feature names from the best model and training data, respectively.")

(let [feature-names (vec feature-names)
      feature-importance (vec feature-importance)
      data (tc/dataset {:vars feature-names
                        :importance feature-importance})
      sorted (tc/order-by data :importance :desc)]
  (-> sorted
      (haclo/layer-bar
       {:=y :vars :=x :importance       ;order-by??
        :=title "Feature Importances"})))

(md "The barplot shows the regressor coefficients, which are proportional to the feature importances. The plot is generated using the Hanami plotting library.")

(answer
 (str "Feature importances:\n"
      (clojure.string/join "\n"
                           (for [[feature importance] (map vector feature-names feature-importance)]
                             (format "%-20s : %.4f ;  " feature importance)))))

(md "This generates a formatted string output of all feature importances, aligning feature names and their corresponding importance values.")

;; Sort features by absolute importance
(def sorted-features
  (->> (map vector feature-names feature-importance)
       (sort-by #(Math/abs (second %)))
       reverse))

(md "This code sorts the features by the absolute value of their importance (coefficient magnitude) in descending order. This is useful because both large positive and large negative coefficients indicate important features in linear models.")

(md "Finally, this code outputs a formatted string of the top 5 most important features based on the absolute value of their coefficients. This provides a quick summary of which features have the largest impact on the model's predictions.")


(answer
 (str "\nTop 5 most important features:\n"
      (clojure.string/join "\n"
                           (for [[feature importance] (take 5 sorted-features)]
                             (format "%-20s : %.4f ;  " feature (float importance))))))

(md
 "### Explanation of feature importances:
 
1. The most important features are those with the largest absolute coefficient values.
2. Positive coefficients indicate that an increase in the feature leads to an increase in the predicted house price, while negative coefficients indicate the opposite.
3. The magnitude of the coefficient represents the feature's relative importance in predicting the house price.
4. Features with near-zero coefficients have little impact on the prediction.

This analysis helps us understand which factors most strongly influence house prices in the Boston housing dataset.")


