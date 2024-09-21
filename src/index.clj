^:kindly/hide-code
(ns index
  (:require 
   [assignments.hw2.utils :refer [question answer md]]
   [scicloj.kindly.v4.kind :as kind])
  (:import [java.time LocalDate]
           [java.time.format DateTimeFormatter]))

(let [formatter (DateTimeFormatter/ofPattern "M/d/yy")
      current-date (str (.format (LocalDate/now) formatter))]
  (md (str "

### Jaryt Salvo
**Date:** **" current-date "**

**Fall 2024 | CS7300 Unsupervised Learning**

*************

This project contains solutions to Homework 2 from the Unsupervised Learning course (CS7300) using Clojure. The primary purpose is to answer the given questions and demonstrate understanding of the concepts. The homework consists of three main questions:

1. Classification with Nearest Neighbor (KNN): Implementing and evaluating KNN on the Iris dataset, including data splitting, elbow method for K selection, and exploration of different distance metrics.

2. Cross Validation for Ridge Regression: Performing Ridge regression with K-fold cross-validation on the Boston Housing dataset, including data loading, model fitting, and feature importance analysis.

3. Writing Questions: Discussing the curse of dimensionality, comparing Ridge and Lasso regression, and providing insights on direct and iterative optimization methods.

The code is organized into different sections corresponding to each homework problem, with detailed explanations of the algorithms and mathematical concepts involved. We utilize Clojure and its associated libraries, such as `scicloj.clay` for rendering, `tablecloth` for data manipulation, and `fastmath` for mathematical operations.

### Key Features of the Implementation:

1. **KNN Classification (Q1):**
   - Data loading and splitting (70/20/10 for train/validation/test)
   - Implementation of the elbow method for optimal K selection
   - Exploration of different distance metrics (Euclidean, Manhattan, Cosine)
   - Model evaluation and performance analysis

2. **Ridge Regression (Q2):**
   - Use of scikit-learn via libpython-clj for model implementation
   - K-fold cross-validation for hyperparameter tuning
   - Feature importance analysis and visualization
   - Model interpretation and equation formulation

3. **Theoretical Concepts (Q3):**
   - In-depth discussion of the curse of dimensionality and mitigation strategies
   - Comparative analysis of Ridge and Lasso regularization techniques
   - Insights into direct and iterative optimization methods

### Technologies and Libraries Used:

- Clojure as the primary programming language
- libpython-clj for Python interoperability (scikit-learn)
- tablecloth for data manipulation
- fastmath for statistical computations
- scicloj.clay for notebook rendering
- Hanamicloth for data visualization

This project demonstrates the application of machine learning concepts using Clojure, showcasing both practical implementations and theoretical understanding of key algorithms in unsupervised learning.

The code in the `src/assignments` folder was rendered with [Clay](https://scicloj.github.io/clay/) and deployed with [Github Pages](https://pages.github.com/).")))
