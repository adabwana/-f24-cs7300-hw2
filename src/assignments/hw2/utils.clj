(ns assignments.hw2.utils
  (:require
   [scicloj.kindly.v4.api :as kindly]
   [scicloj.kindly.v4.kind :as kind]
   [tablecloth.api :as tc]
   [tech.v3.datatype.functional :as dtf]
   [uncomplicate.neanderthal
    [core :refer [col entry nrm2 mv mrows ncols
                  scal axpy copy mm dia trans]]
    [native :refer [dge]]
    [linalg :refer [ev! svd]]]))

^:kindly/hide-code
(kind/md "## Utils")


;; Formatting code
(def md (comp kindly/hide-code kind/md))
(def question (fn [content] ((comp kindly/hide-code kind/md) (str "## " content "\n---"))))
(def sub-question (fn [content] ((comp kindly/hide-code kind/md) (str "#### *" content "*"))))
(def sub-sub (fn [content] ((comp kindly/hide-code kind/md) (str "***" content "***"))))
(def formula (comp kindly/hide-code kind/tex))
(def answer 
  (fn [content] 
    (kind/md 
     (str "> <span style=\"color: black; font-size: 1.5em;\">**" content "**</span>"))))



;; normalize/standardize
(defn tc-col->vec
  [data col]
  (vec (col (tc/select-columns data col))))


(defn normalize-column
  "Normalize a column using min-max normalization"
  [col]
  (let [min-val (apply min col)
        max-val (apply max col)]
    (map #(/ (- % min-val) (- max-val min-val)) col)))


;; TODO: in neanderthal, use dge/mean and dge/std-dev
(defn standardize-column
  "Standardize a column using z-score normalization"
  [col]
  (let [mean (dtf/mean col)
        std (dtf/standard-deviation col)]
    (map #(/ (- % mean) std) col)))


(defn apply-to-all-columns
  "Apply a function to all columns of a dataset"
  [data func]
  (reduce (fn [acc col-name]
            (tc/add-column acc col-name (func (col-name data))))
          (tc/dataset)
          (tc/column-names data)))

(let [data (tc/dataset {:age [20 30 35 40 50]
                        :income [30000 40000 59000 55000 90000]
                        :credit-score [650 700 750 800 850]})

      normalized-data (apply-to-all-columns data normalize-column)]
  (tc/dataset normalized-data))


;; center
(defn matrix->dataset
  "Transforms a Neanderthal matrix into a Tablecloth dataset.
   Columns are named x1 to xp, where p is the number of columns in the matrix."
  [matrix]
  (let [rows (mrows matrix)
        cols (ncols matrix)
        column-names (mapv #(keyword (str "x" (inc %))) (range cols))
        data (for [i (range rows)]
               (for [j (range cols)]
                 (entry matrix i j)))]
    (tc/dataset (map (fn [row] (zipmap column-names row)) data))))


(defn dataset->matrix
  "Converts a Tablecloth dataset to a Neanderthal matrix."
  [dataset]
  (let [X (tc/rows dataset :as-seqs)]                       ;opt ':as-seq' is default
    (dge (count X) (count (first X))                        ;{:layout :column} is default
         (flatten X) {:layout :row})))

(defn center-data
  "Centers the data by subtracting the mean of each column.
   Parameters:
   - data: Neanderthal matrix to be centered"
  [data]
  (let [mean-vec (dge 1 (ncols data) (map #(/ (reduce + %) (mrows data)) (trans data)))
        centered-data (axpy -1 (mm (dge (mrows data) 1 (repeat (mrows data) 1)) mean-vec) data)]
    centered-data))


(defn compute-covariance-matrix
    "Computes the covariance matrix of the centered data.
    Parameters:
    - centered-data: Centered Neanderthal matrix"
    [centered-data]
    (mm (trans centered-data) centered-data))

^:kindly/hide-code
(comment ;; TODO: eigenvectors for pca
  (let [data (dge 3 3 [[1 2 3] [11 9 4] [6 3 5]])
        centered-data (center-data data)
        cov-matrix (compute-covariance-matrix centered-data)
        n (mrows cov-matrix)
        eigenvectors (dge n n)  ; Ensure eigenvectors is sized to match cov-matrix
        eigenvalues (ev! cov-matrix eigenvectors)]
    (println "Eigenvalues:")
    (println eigenvalues)
    (println "\nEigenvectors:")
    (println eigenvectors))

  (defn compute-eigenvectors
    "Computes and sorts eigenvectors by descending eigenvalues.
   Parameters:
   - cov-matrix: Covariance matrix"
    [cov-matrix]
    (let [n (mrows cov-matrix)
          eigenvectors (dge n n)  ; Ensure eigenvectors is sized to match cov-matrix
          eigenvalues (ev! (copy cov-matrix) eigenvectors)
          sorted-indices (reverse (sort-by #(entry eigenvalues % 0) (range n)))
          sorted-eigenvectors (dge n n
                                   (flatten (map #(col eigenvectors %) sorted-indices)))]
      sorted-eigenvectors))



  (let [data (dge 3 3 [[1 2 3] [4 5 6] [7 8 9]])
        centered-data (center-data data)
        cov-matrix (compute-covariance-matrix centered-data)
        sorted-eigenvectors (compute-eigenvectors cov-matrix)]
    (println "Sorted eigenvectors:")
    (println sorted-eigenvectors))


  (comment
    (defn compute-eigenvectors
      "Computes and sorts eigenvectors by descending eigenvalues.
   Parameters:
   - cov-matrix: Covariance matrix"
      [cov-matrix]
      (let [n (mrows cov-matrix)
            a-copy (copy cov-matrix)
            eigenvalues (dge n 2)
            eigenvectors (dge n n)
            _ (ev! a-copy eigenvectors)
        ; Sort based on the magnitude of complex eigenvalues
            sorted-indices (reverse (sort-by #(+ (Math/pow (entry eigenvalues % 0) 2)
                                                 (Math/pow (entry eigenvalues % 1) 2))
                                             (range n)))
            sorted-eigenvectors (dge n n
                                     (flatten (map #(col eigenvectors %) sorted-indices)))]
        {:eigenvalues eigenvalues
         :eigenvectors sorted-eigenvectors}))

    (let [data (dge 3 3 [1 2 3 4 5 6 7 8 9])
          centered-data (center-data data)
          cov-matrix (compute-covariance-matrix centered-data)
          {:keys [eigenvalues eigenvectors]} (compute-eigenvectors cov-matrix)]
      (println "Eigenvalues:")
      (println eigenvalues)
      (println "\nSorted eigenvectors:")
      (println eigenvectors))


    (defn project-data
      "Projects the centered data onto the first n principal components.
   Parameters:
   - centered-data: Centered Neanderthal matrix
   - sorted-eigenvectors: Sorted eigenvectors
   - n: Number of principal components to use"
      [centered-data sorted-eigenvectors n]
      (mm centered-data (trans (dge (ncols sorted-eigenvectors) n
                                    (flatten (map #(col sorted-eigenvectors %) (range n)))))))

    (defn perform-pca
      "Performs PCA on the given dataset and returns the projected data.
   Parameters:
   - dataset: Tablecloth dataset
   - n-components: Number of principal components to use"
      [dataset n-components]
      (let [data (dataset->matrix dataset)
            centered-data (center-data data)
            cov-matrix (compute-covariance-matrix centered-data)
            sorted-eigenvectors (compute-eigenvectors cov-matrix)
            pca-result (project-data centered-data sorted-eigenvectors n-components)]
        pca-result))

    (defn pca-example []
      (let [dataset (tc/dataset {:age [25 30 35 40 45]
                                 :income [30000 45000 60000 75000 90000]
                                 :credit-score [650 700 750 800 850]})
            pca-result (perform-pca dataset 2)]
        (println "Original data:")
        (println dataset)
        (println "\nPCA result (first two components):")
        (println pca-result)))))