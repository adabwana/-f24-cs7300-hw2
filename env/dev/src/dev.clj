(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "Homework 2"}
    :subdirs-to-sync     ["notebooks" "data"]
    :source-path         ["src/index.clj"
                          ;; "src/assignments/hw2/utils.clj"
                          "src/assignments/hw2/q1.clj"
                          "src/assignments/hw2/q2.clj"
                          "src/assignments/hw2/q3.clj"]
    :base-target-path    "docs"
    :clean-up-target-dir true}))

(comment
  (build))
