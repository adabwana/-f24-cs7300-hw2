(ns my-py-clj.config
  (:require [libpython-clj2.python :as py]))

(py/initialize!)

(comment
  (py/initialize! :python-executable ".venv/bin/python3"
                  :library-path ".venv/lib/python3.11/site-packages"
                  :python-path ".venv/bin/python3"))