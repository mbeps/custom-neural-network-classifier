**6CCS3ML1 (Machine Learning)**  
**Coursework 1**  
**(Version 1.6)**  

---

## 1. Overview  
For this coursework, you will have to implement a classifier. You will use this classifier in some code that has to make a decision. The code will be controlling Pacman, in the classic game, and the decision will be about how Pacman chooses to move. Your classifier probably won’t help Pacman to make particularly good decisions (I will be surprised if it helps Pacman win games, my version certainly didn’t), but that is not the point. The point is to write a classifier and use it.

No previous experience with Pacman (either in general, or with the specific UC Berkeley AI implementation that we will use) is required.

This coursework is worth **10% of the marks** for the module.

**Note:** Failure to follow submission instructions will result in a deduction of **10% of the marks** you earn for this coursework.

---

## 2. Getting started  

### 2.1 Start with Pacman  
The Pacman code that we will be using for the coursework was developed at **UC Berkeley** for their AI course. The homepage for the **Berkeley AI Pacman projects** is here:

[http://ai.berkeley.edu/](http://ai.berkeley.edu/)

**Note:** We will **not** be doing any of their projects. Note also that the code only supports **Python 3**, so that is what we will use.

#### Steps to follow:
1. **Download:**
   - `pacman-cw1.zip` from KEATS.
2. **Save** that file to your account at **KCL** (or to your own computer).
3. **Unzip** the archive.
   - This will create a folder `pacman`
4. **Run Pacman:**
   - Open a command line, navigate to `pacman` directory, and type:
     ```
     python3 pacman.py
     ```
   - This will open up a window displaying Pacman.
   - The default mode is for keyboard control (use arrow keys).

**Note:** Playing Pacman is not the objective here—don’t worry if you have trouble controlling Pacman using the keys. You only need to run the code to complete the coursework. If there is an error, get help.

### 2.2 Code to control Pacman  
To control Pacman programmatically, check the file **sampleAgents.py**. This file contains simple agents to control Pacman. You can run an agent by executing:
```  
python3 pacman.py --pacman RandomAgent
```

This runs a **RandomAgent** which selects actions at random.

#### Key Points:
- You execute an agent using `--pacman` followed by the name of a **Python class**.
- Pacman looks for this class in files ending in `*Agents.py`.
- If the class is missing, you will get an error.
- The **getAction()** function in the agent determines Pacman’s moves.

### 2.3 Towards a classifier  
For this coursework, work from the skeleton code in `pacman-cw1`. The file to modify is **classifier.py**, which is used in **classifierAgents.py**.

**Important Notes:**
1. The class **Classifier** (in `classifier.py`) and the class **ClassifierAgent** (in `classifierAgents.py`) must **not** be renamed.
2. Your submission will be tested using **ClassifierAgent**.
3. The **ClassifierAgent** reads data from `good-moves.txt` and converts it into `data` and `target` arrays.
4. Your agent must be run using:
   ```
   python3 pacman.py --pacman ClassifierAgent
   ```

---

## 3. What you have to do (and what you aren’t allowed to do)  

### 3.1 Write some code  
Your task is to write a classifier in **classifier.py** that uses `good-moves.txt` to control Pacman.

#### Restrictions:
1. You can **only** access **api.getFeatureVector(state)**, which provides a feature vector of 1s and 0s.
2. Your classifier must be trained using **self.data** and **self.target**.
3. You **can** use an external classifier (e.g. from scikit-learn), but writing your own will earn **more marks**.
4. If you write your own classifier, a simple **1-nearest neighbour classifier** is acceptable, but more sophisticated approaches will score higher.
5. Your code **must not crash**—Pacman must either win or lose but not fail due to errors.

### 3.2 Things to know  
- **good-moves.txt** contains feature vectors and corresponding moves.
- You may want to create custom training data using **TraceAgent**.
  ```
  python3 pacman.py --p TraceAgent
  ```
  - This logs moves and feature vectors to `moves.txt`.

### 3.3 Limitations  
1. Code must be in **Python 3**.
2. Code will be tested in **Anaconda Python 3** with `scikit-learn` and `numpy`.
3. Your code **must only interact** with Pacman through `api.py`.
4. Do **not** modify any files except **classifier.py**.
5. Plagiarism rules apply—do **not** copy code from others **without credit**.

---

## 4. What you have to hand in  
You must submit a **single ZIP file** containing **only one file**:
- **classifier.py**

**ZIP file naming format:**
```
cw1-<lastname>-<firstname>.zip
```

Your code will be run using:
```
python3 pacman.py -p ClassifierAgent
```
Make sure that your **ClassifierAgent** and **Classifier** classes exist as per the skeleton provided.

**Do not include the entire pacman-cw1 folder.**

Submissions that do not follow these instructions will lose marks.

---

## 5. How your work will be marked  
Marks are based on three criteria:

### (a) Functionality  
- Your classifier must work when tested against a **clean** version of `pacman-cw1`.
- It must run using:
  ```
  python3 pacman.py --p ClassifierAgent
  ```
- Your code must include a classifier.
- More sophisticated classifiers earn **more marks**.

### (b) Style  
- Your code should follow **good coding practices**.
- It should only interact with **api.py** and use allowed environment information.

### (c) Documentation  
- Your code must be **well-commented**.
- If we cannot understand your logic, you **will lose marks**.

---

A **marksheet** and **FAQs** are available on KEATS.

