(ns assignments.hw2.q1
  (:require
   [my-py-clj.config :refer :all]
   [assignments.hw2.utils :refer :all]
   [fastmath.stats :as stats]
  ;;  [libpython-clj2.python :refer [py.-]]
  ;;  [scicloj.sklearn-clj.metamorph :as sklearn-mm]
   [scicloj.hanamicloth.v1.api :as haclo]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as mm]
   [scicloj.metamorph.ml.classification :as mlc]
   [scicloj.metamorph.ml.gridsearch :as grid]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.sklearn-clj :as sklearn-clj]
   [scicloj.sklearn-clj.ml]                                ;; registers all models
   [tablecloth.api :as tc]
   [tech.v3.dataset.metamorph :as dsm]
   [tech.v3.dataset.modelling :as ds-mod]))


(question "Question 1")

(sub-question "Q1: Classification with Nearest Neighbor (30 Points)")
(md 
 "For this question, you will need to perform KNN on the famous Iris data set. You are required to use Scki-learn to completion this question. The data is stored in a csv file and it can be downloaded from Canvas. For the description of the data set, you can visit: [Wikipedia iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).")

(sub-sub "1) Load the data and split it into train, valid, and test (70/20/10).")

(defonce iris (-> "data/A1Q1_Data.csv"
                  (tc/dataset {:key-fn (fn [colname]
                                         (-> colname    ;kabab-case keyword
                                             (clojure.string/replace #"\.|\s" "-")
                                             clojure.string/lower-case
                                             keyword))})
                  (ds-mod/set-inference-target :variety)))


(def response :variety)
(def regressors
  (tc/column-names iris (complement #{response})))


(let [data (-> iris
               (tc/split->seq :holdout {:seed 123 :ratio 0.9}))
      test-data (-> data first :test)
      train-val-data (-> data first :train
                         (tc/split->seq :holdout {:seed 123}))
      train-data (-> train-val-data first :train)
      val-data (-> train-val-data first :test)]
  (def test-data test-data)
  (def train-data train-data)
  (def val-data val-data))

(md
 "`tc/split->seq` is a function that splits a dataset into two or more subsets. In this case, it's dividing the dataset into a test set and a training set. The `:holdout` option specifies that we want to split the dataset into two subsets, while the `:ratio` option determines the proportion of the dataset to include in the training set. 
  
 With this 90/10 split, we can further divide the training set into training and validation sets to tune our hyperparameters. The test set is already accessible in the data variable by calling `first`, which indicates the first element (map) of the sequence. The `:test` key in this map represents the 10% of the data set aside for testing, as specified in the `tc/split->seq` function call.")


(sub-sub "2) Write a function that uses the elbow method to select the value for K. You can set the range for K as (1, 15).")

^:kindly/hide-code
(comment
  (sort (mm/model-definition-names))

  (mm/hyperparameters :sklearn.classification/k-neighbors-classifier))


(defn create-pipeline [params]
  (morph/pipeline
   (dsm/categorical->number [response])
   (dsm/set-inference-target response)
   {:metamorph/id :model}
   (mm/model (merge {:model-type :sklearn.classification/k-neighbors-classifier}
                    params))))

(md "This function creates a pipeline for the KNN model, converting categorical data to numbers, setting the inference target, and creating the model with given parameters.")

(defn generate-hyperparams []
  (grid/sobol-gridsearch
   {:n-neighbors (grid/linear 1 15 15 :int32)
    :weights     (grid/categorical ["distance"])
    :metric      (grid/categorical ["euclidean" "manhattan" "cosine"])}))

(md "This function generates hyperparameters for the KNN model using Sobol sequence for efficient space exploration, including neighbors (1-15), weights, and distance metrics.")

(grid/sobol-gridsearch
 {:n-neighbors (grid/linear 1 2 2 :int32)
  :metric      (grid/categorical ["euclidean" "manhattan"])})


(defn evaluate-model [pipelines data-seq]
  (mm/evaluate-pipelines
   pipelines
   data-seq
   stats/cohens-kappa
   :accuracy
   {:other-metrices
    [{:name :mathews-cor-coef :metric-fn stats/mcc}
     {:name :accuracy :metric-fn loss/classification-accuracy}]
    :return-best-pipeline-only        false
    :return-best-crossvalidation-only true}))

(md "This function evaluates the model pipelines using various metrics like Cohen's kappa, Matthews correlation coefficient, and accuracy.")

(defn process-results [evaluations]
  (->> evaluations
       flatten
       (map #(hash-map
              :summary (mm/thaw-model (get-in % [:fit-ctx :model]))
              :fit-ctx (:fit-ctx %)
              :timing-fit (:timing-fit %)
              :metric ((comp :metric :test-transform) %)
              :other-metrices ((comp :other-metrices :test-transform) %)
              :params ((comp :options :model :fit-ctx) %)
              :lookup-table (get-in % [:fit-ctx :model :target-categorical-maps :variety :lookup-table])
              :pipe-fn (:pipe-fn %)))
       (sort-by :metric)
       reverse))

(md "This function processes the evaluation results, extracting relevant information and sorting the results by metric score in descending order.")

(defn elbow-method [train-data val-data]
  (let [pipelines (map create-pipeline (generate-hyperparams))
        evaluations (evaluate-model pipelines [{:train train-data :test val-data}])]
    (process-results evaluations)))

(md "This function implements the elbow method by creating pipelines with different hyperparameters, evaluating them, and processing the results to find the optimal K value.")

(def elbow-results
  (elbow-method train-data val-data))

(md "This line applies the elbow method to the training and validation data, storing the results as `elbow-results`.")

(count elbow-results)


(def results-dataset
  (->> elbow-results
       (map (fn [result]
              (let [k (get-in result [:params :n-neighbors])
                    dist-metric (get-in result [:params :metric])
                    kappa (:metric result)
                    other-metrics (:other-metrices result)
                    mcc (-> (filter #(= (:name %) :mathews-cor-coef) other-metrics)
                            first
                            :metric)
                    accuracy (-> (filter #(= (:name %) :accuracy) other-metrics)
                                 first
                                 :metric)]
                {:k        k
                 :metric   dist-metric
                 :kappa    kappa
                 :mcc      mcc
                 :accuracy accuracy})))
       (tc/dataset)))

(md "This code processes the elbow results, extracting key metrics (k, distance metric, kappa, MCC, accuracy) and creates a dataset for easier analysis and visualization.")

(let [data (tc/select-rows results-dataset (comp #(= % "euclidean") :metric))]
  (-> data
      (haclo/layer-line {:=x :k :=y :kappa})
      (haclo/layer-point {:=x         :k :=y :kappa
                          :=mark-size 50})))

(md "This code creates a plot of the elbow method results for the Euclidean distance metric, showing how kappa changes with different K values.")


^:kindly/hide-code
(comment
  (let [data (tc/select-rows results-dataset (comp #(= % "euclidean") :metric))]
    (-> data
        (haclo/layer-line {:=x :k :=y :mcc})
        (haclo/layer-point {:=x         :k :=y :mcc
                            :=mark-size 50}))))

^:kindly/hide-code
(answer "Looking at our elbow plot, we can see that the elbow dips down at $K=8$. Therefore, we want a K before 8. To me, 1 is too small and even numbers won't necessarily have a majority in a vote. I'd say good choices are 3, 5, or 7. I'll choose 5 because it's the middle value.")


(sub-sub "3) Explore different distance metrics and repeat part 2 with another distance metric. In the report, justify the value of K and the distance metric.")


(let [data (tc/select-rows results-dataset (comp #(= % "manhattan") :metric))]
  (-> data
      (haclo/layer-line {:=x :k :=y :mcc})
      (haclo/layer-point {:=x         :k :=y :mcc
                          :=mark-size 50})))


(let [data (tc/select-rows results-dataset (comp #(= % "cosine") :metric))]
  (-> data
      (haclo/layer-line {:=x :k :=y :mcc})
      (haclo/layer-point {:=x         :k :=y :mcc
                          :=mark-size 50})))

(md "
**Justification for K value and distance metric:**

*1. Euclidean distance (default):*
     
   - Optimal K: $[3, 5, 7]$
   - This metric is suitable for continuous features and assumes all features contribute equally.
   - It works well when the relationship between features is linear.

*2. Manhattan distance:*
     
   - Optimal K: $[3, 5]$
   - This metric is less sensitive to outliers compared to Euclidean distance.
   - It's particularly useful when features are on different scales or when dealing with high-dimensional data.
     
*3. Cosine distance:*
     
   - Optimal K: $[6, 10, 12, 14, 15]$
   - This metric is useful for high-dimensional data where feature scaling is important.
   - It's particularly useful when the angle between data points is more important than their magnitude.

The choice between these metrics depends on the specific characteristics of the Iris dataset:

- If the features are on similar scales and have a roughly linear relationship, Euclidean distance might be preferred.
- If there are potential outliers or the features are on different scales, Manhattan distance could be more appropriate.
- If the data is high-dimensional and feature scaling is important, cosine distance might be the best choice.")


^:kindly/hide-code
(answer "The optimal K value balances between overfitting (low K) and underfitting (high K). The choice of distance metric depends on the specific characteristics of the dataset. Euclidean distance is a fine choice for this dataset for reasons explained above as well as being the easiest to understand. As such, I'd probably choose $K=5$ using the Euclidean distance metric.")


(sub-sub "4) Train the KNN model with the optimal K value and chosen distance metric on the combined train and validation sets.")


(def train-val-data-bootstrapped
  (-> train-data
      (tc/concat val-data)
      (tc/split->seq :bootstrap {:seed 123 :repeats 25})))

(md "This code combines the training and validation data, then creates 25 bootstrap samples for robust model evaluation.")

(def final-model
  (let [pipelines (map create-pipeline [{:n-neighbors 5
                                         :weights     "distance"
                                         :metric      "euclidean"}])
        evaluations (evaluate-model pipelines train-val-data-bootstrapped)]
    (process-results evaluations)))

(md "Here we define the final model using the chosen hyperparameters (5 neighbors, Euclidean distance) and evaluate it on the bootstrapped data.")

(-> final-model first :lookup-table)

(md "This line retrieves the lookup table from the final model, which maps categorical labels to numerical values.")


(sub-sub "5) Evaluate the model on the test set and report the accuracy.")


^:kindly/hide-code
(comment
  (defn preds
    [model]
    (-> test-data
        (morph/transform-pipe
         (-> model first :pipe-fn)
         (-> model first :fit-ctx))
        :metamorph/data
        :variety
        (->> (map #(long %))
             vec)))

  (defn actual
    [model]
    (-> test-data
        (morph/fit-pipe
         (-> model first :pipe-fn))
        :metamorph/data
        :variety
        vec))

  (defn actual
    [model]
    (let [lookup-table (-> model first :lookup-table)]
      (-> test-data
          (morph/fit-pipe
           (-> model first :pipe-fn))
          :metamorph/data
          :variety
          (->> (map #(get lookup-table %))
               vec)))))


(defn actual
  [model]
  (-> test-data :variety vec))

(defn preds
  [model]
  (let [lookup-table (-> model first :lookup-table)
        reverse-lookup (zipmap (vals lookup-table) (keys lookup-table))]
    (-> test-data
        (morph/transform-pipe
         (-> model first :pipe-fn)
         (-> model first :fit-ctx))
        :metamorph/data
        :variety
        (->> (map #(get reverse-lookup (long %)))
             vec))))

(md "These functions extract the actual labels from the test data and generate predictions using the trained model, respectively.")

(defn evaluate-predictions
  "Evaluates predictions against actual labels, returns confusion map and metrics."
  [preds actual]
  (let [conf-map (mlc/confusion-map->ds (mlc/confusion-map preds actual :none))
        accuracy (loss/classification-accuracy preds actual)
        kappa (stats/cohens-kappa preds actual)
        mcc (stats/mcc preds actual)]
    {:confusion-map conf-map
     :accuracy      (format "%.4f" accuracy)
     :cohens-kappa  (format "%.4f" kappa)
     :mcc           (format "%.4f" mcc)}))

(md "`evaluate-predictions` calculates various performance metrics including accuracy, Cohen's kappa, and Matthews correlation coefficient, along with a confusion matrix. At last, we generate predictions on the test set, compare them with the actual labels, and compute the evaluation metrics to assess the model's performance.")

(let [preds (preds final-model)
      actual (actual final-model)]
  (evaluate-predictions preds actual))

(answer (str "Accuracy on the Iris test set: "
             (get-in (let [preds (preds final-model)
                           actual (actual final-model)]
                       (evaluate-predictions preds actual))
                     [:accuracy])))

(sub-sub "6) Provide a brief analysis of the results in your report.")


^:kindly/hide-code
(md 
 "### Analysis of Results:

#### 1. Data Split: 
  We used a 70/20/10 split for train/validation/test, which is a common practice. This split provides enough data for training while reserving sufficient data for validation and testing. However, the Iris dataset is small, where the 10% holdout is just 15 samples.

#### 2. Elbow Method: 
  This approach effectively determined the optimal K value, balancing between model complexity and performance, between underfitting and overfitting.

#### 3. Distance Metrics:
  Comparison of Euclidean and Manhattan distances revealed metric-dependent optimal K values, underscoring the importance of metric selection in KNN.

#### 4. Final Model:
  We selected Euclidean distance with K = 5, based on the elbow method results. This choice aims to balance model simplicity with performance.

#### 5. Model Performance:
  The final accuracy of $0.9333$ on the test set demonstrates strong generalization to unseen data, validating our K and distance metric choices.

#### 6. Limitations and Future Work:
  
  - Explore additional distance metrics and feature scaling techniques for potential performance improvements.
  - Implement k-fold cross-validation for more robust performance estimation.
  - Conduct feature importance analysis to identify key iris characteristics for classification.
  - Consider testing the model on a larger, more diverse dataset to assess its broader applicability.")


^:kindly/hide-code
(comment
  (def train-ds
    (-> (tc/dataset {:x1 [1 1 2 2]
                     :x2 [1 2 2 3]
                     :y  [6 8 9 11]})
        (ds-mod/set-inference-target :y)))

  (def test-ds
    (->
     (tc/dataset {:x1 [3]
                  :x2 [5]
                  :y  [0]})
     (ds-mod/set-inference-target :y)))

  (def lin-reg
    (sklearn-clj/fit train-ds :sklearn.neighbors :k-neighbors-classifier))

  ;; Call predict with new data on the estimator
  (sklearn-clj/predict test-ds lin-reg [:y]))
