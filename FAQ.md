**6CCS3ML1 (Machine Learning)**  
**Coursework 1 FAQ**  
**(Version 1.3)**  

---

### Q: Running `python3 pacman.py -p ClassifierAgent` with a clean copy of the downloaded file throws an exception: "The agent ClassifierAgent is not specified in any *Agents.py." Looking at the code in pacman.py, it seems trying to import classifierAgent is throwing an ImportError. How can we resolve this?  
**A:** Ensure you run everything in the environment we have set up in the lab machines (see coursework document). This is the standard Anaconda Python 3 distribution, with scikit-learn also installed. If you run this on your own machine, ensure you use the same environment. For the error, see also Section 2.2 of the coursework document.  

### Q: "Code for a sophisticated classifier included (i.e. student has created a sophisticated classifier, not just called one from the library.) (20 marks)" What classifiers that we have learnt are considered sophisticated?  
**A:** We are assessing your ability to identify sophisticated models, so we cannot answer this. However, we do not consider the following to be sophisticated: naive bayes, knn, and ensembles of these.  

### Q: Let’s say a naive bayes algorithm using scikit-learn (and/or an implementation made by the student alone has made the algorithm more robust) has been implemented that is somewhat sophisticated. What would this return as a final mark? Let’s assume the code is well written and documented (extra 20 marks). Could this result in 60 marks awarded?  
**A:** We will not assess your coursework in any way at all until it has been submitted and the coursework deadline date has passed.  

### Q: I have implemented a Naive Bayes classifier and combined 2 other sklearn classifiers to make an Ensemble Classifier. The code is below 100 lines, but I believe I have satisfied all the requirements. I have written a classifier, and it is sophisticated. Is it sophisticated enough?  
**A:** We do not give feedback before you submit your work.  

### Q: Are we allowed to implement a classifier that has not been covered in lectures?  
**A:** Yes, that is up to you.  

### Q: Should we just train in our own environment and load the parameters to the classifier (i.e., do not show the training process)?  
**A:** No, you need to include everything.  

### Q: "Runs on a different training set" – does this simply mean different versions of `good-moves.txt`, and not different format of the input data?  
**A:** Correct.  

### Q: I am confused about how we are supposed to show that we used different training data if we are only submitting the `classifier.py` file?  
**A:** You are not supposed to show this explicitly. Ensure that your code runs on different training sets (we have given you code to generate additional data). We will be testing your code with our own training files to check whether it works.  

### Q: I am concerned about the runtime when a large dataset is provided. Will this be counted against me?  
**A:** We will test your code on similarly sized datasets as the one you have been given, so that should not be an issue. Once your model is trained, however, it should be quick in making decisions.  

### Q: Do feature vectors in `good-moves.txt` represent a single continuous Pacman game run, or are they randomly selected moves? Will the test data follow the same pattern?  
**A:** That information is not relevant to the task. If helpful, the best answer is "probably." The task is to build a classifier, not a sequential model.  

### Q: The feature vector length differs for different grids (e.g., 17 for smallGrid, 25 for mediumClassic). Should our agent work on any grid or only those having 25 numbers as features?  
**A:** The feature vector length depends on the layout. All datasets used for testing will be based on `mediumClassic`, so they will match the `good-moves.txt` format.  

### Q: Can we use functions from numpy (e.g., mean, zeros)? Will we lose marks for using these?  
**A:** The only limitations are stated in Section 3.3 of the coursework document.  

### Q: Can we use non-classifier functions from scikit-learn (e.g., train-test split)?  
**A:** You can use external library functions, but you get more marks for writing more code yourself. The more functionality you implement, the more credit you receive.  

### Q: How should we cite sources for our own classifier implementation? Is a README allowed?  
**A:** Add citations and descriptions in the Python file itself, as it is easier to mark.  

### Q: Will using existing classifier algorithms lower my coursework grade since it is "not entirely my own work"?  
**A:** You get credit for what you implement. Writing an existing classifier yourself gains more marks than using one from a library. Creating an ensemble combining existing classifiers earns more than just calling a library function. Writing a whole new classifier earns even more marks.  

### Q: My classifier has low accuracy despite being implemented correctly. Will this affect my marks?  
**A:** Accuracy is not assessed. Section 3.1(e) states: "Losing a game is not failing. We only care that your code successfully uses a classifier to decide what to do."  

### Q: Should we check if the action chosen by the classifier is legal to avoid Pacman getting stuck or crashing?  
**A:** That is up to you. Your agent must not crash, so test your code carefully to ensure it only returns legal moves.

