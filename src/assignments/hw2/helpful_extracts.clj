(ns utils.helpful-extracts
  (:require
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]))

(defn evaluate-pipe [pipe data]
  (ml/evaluate-pipelines
    pipe
    data
    stats/omega-sq
    :accuracy
    {:other-metrices                   [{:name :mae :metric-fn ml/mae}
                                        {:name :rmse :metric-fn ml/rmse}]
     :return-best-pipeline-only        false
     :return-best-crossvalidation-only true}))

(defn sort-map-by-key [key m]
  (let [sorted-indices (map-indexed (fn [idx val] [idx val]) (m key))
        sorted-indices (sort-by second sorted-indices)
        sorted-keys (map first sorted-indices)]
    (reduce (fn [sorted-map k]
              (assoc sorted-map k (map #(get (m k) %) sorted-keys)))
            {}
            (keys m))))

(defn best-models [eval]
  (->> eval
       flatten
       (map
         #(hash-map :summary (ml/thaw-model (get-in % [:fit-ctx :model]))
                    :fit-ctx (:fit-ctx %)
                    :timing-fit (:timing-fit %)
                    :metric ((comp :metric :test-transform) %)
                    :other-metrices ((comp :other-metrices :test-transform) %)
                    :other-metric-1 ((comp :metric first) ((comp :other-metrices :test-transform) %))
                    :other-metric-2 ((comp :metric second) ((comp :other-metrices :test-transform) %))
                    :params ((comp :options :model :fit-ctx) %)
                    :pipe-fn (:pipe-fn %)))
       (sort-by :metric)))

(defn extract-params [model num]
  (let [numbers (range num)]
    (for [n numbers]
      (-> model (nth n) :params))))

(defn- extract-compute-time [model num]
  (let [numbers (range num)]
    (for [n numbers]
      (-> model (nth n) :timing-fit))))

(defn- extract-metric-adjr2s [model num]
  (let [numbers (range num)]
    (for [n numbers]
      (-> model (nth n) :metric))))

(defn- extract-metric-maes [model num]
  (let [numbers (range num)]
    (for [n numbers]
      (-> model (nth n) :other-metrices first :metric))))

(defn- extract-metric-rmses [model num]
  (let [numbers (range num)]
    (for [n numbers]
      (-> model (nth n) :other-metrices second :metric))))

; Here is that public function leveraging the above private functions binding them to variables of interest.

(defn eval-maps [model num]
  (let [lambdas (extract-params model num)
        comp-time (map #(assoc {} :compute-time-ns %) (extract-compute-time model num))
        adjr2 (map #(assoc {} :adj-r2 %) (extract-metric-adjr2s model num))
        mae (map #(assoc {} :mae %) (extract-metric-maes model num))
        rmse (map #(assoc {} :rmse %) (extract-metric-rmses model num))]
    (map #(conj %1 %2 %3 %4 %5) lambdas comp-time adjr2 mae rmse)))

(defn model->ds
  ([data]
   (let [ks (keys (first data))]
     (ds/dataset (zipmap ks (apply map vector (map (apply juxt ks) data))))))
  ([model num]
   (let [data (eval-maps model num)
         ks (keys (first data))]
     (ds/dataset (zipmap ks (apply map vector (map (apply juxt ks) data)))))))

(defn coefs-vs-lambda [data pipeline]
  (flatten
    (map
      (fn [lambda]
        (let [fitted (ml/fit-pipe data (pipeline {:lambda (double lambda)}))
              model-instance (-> fitted :model (ml/thaw-model))
              predictors (map #(first (.variables %))
                              (seq (.. model-instance formula predictors)))]
          (map
            #(hash-map :log-lambda (Math/log10 lambda)
                       :coefficient %1
                       :predictor %2)
            (-> model-instance .coefficients seq)
            predictors)))
      (range 1 100000 100))))

(comment

  ;; ### Plots of ridge and lasso coefficients vs lambdas
  (-> (coefs-vs-lambda liver-disease ridge-pipe-fn)
      (hanami/plot ht/line-chart
                   {:X     "log-lambda" :XSCALE {:zero false}
                    :Y     "coefficient" :YSCALE {:zero false}
                    :COLOR "predictor" :TITLE "Ridge"}))

  (-> (coefs-vs-lambda liver-disease lasso-pipe-fn)
      (hanami/plot ht/line-chart
                   {:X     "log-lambda" :XSCALE {:zero false}
                    :Y     "coefficient" :YSCALE {:zero false}
                    :COLOR "predictor" :TITLE "Lasso"})))