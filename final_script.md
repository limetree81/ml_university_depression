

Slide 15: Supervised Learning SetupScript:
"Now, let’s look at the setup for our Supervised Learning phase. we used a Random Forest Classifier. We split our data into 80% for training and 20% for testing to predict the "Depression" target variable. Initially, the result looked impressive with a 90% Accuracy. However, I had a serious doubt: looking at our target distribution, 90% of the students in this dataset are not depressed. This means that if I simply predict everyone is 'fine,' the model would still be 90% accurate. We needed a better way to measure if the model was actually learning which is..."

Slide 16: Decoding the Confusion Matrix & MetricsScript:
"A Confusion Matrix. Let's define the four quadrants you see here:TN (True Negative): We correctly predicted a healthy student as 'False'.TP (True Positive): We correctly caught a student who actually has depression. FP (False Positive): we predicted depression, but the actually fine.FN (False Negative): A student is depressed, but we predicted they were fine.From these, we derive our three key metrics:Precision: Calculated as $TP / (FP + TP)$. This tells us: 'Of all the students we flagged as depressed, how many actually were?'Recall: Calculated as $TP / (FN + TP)$. This asks: 'Of all the students who truly have depression, how many did we successfully catch?'F1-Score: This is the harmonic mean of Precision and Recall. It’s a single score that balances both. I'm using macro average f1-score for both true case and false case. I know it's kinda hard to understand all of these features in a moment. but just remember this. Precision is yellow divided by pink and yello. recall is green divided by green and yellow. f1 score is overall score. and for all of them, the higher the number the better the prediction."

Slide 17: Initial Performance AnalysisScript:
"And this is the result of our prediction. It was true that the model predicted most of the case as False. This results in the table below: a Precision of 0.436 and a Recall of only 0.017. I had to improve the prediction. I tried linear model to prevent overfitting."

Slide 18: Overfitting and Logistic RegressionScript:
"I used Logistic Regrssion for prediction. However, look at the matrix now. The model predicted False for every single case of depression. This resulted in a Recall of 0. I thought this is the limit of linear model. and I switched back to Random Foreset Classifier."

Slide 19: Adjusting the ThresholdScript:
"Since the model was too 'conservative,' I manually adjusted the classification threshold to 0.1, and now the result seems rather balanced.. or is it? True Positives jumped from 33 to 1,411.As the table shows, our Recall is now 0.71! But there's a trade-off. Our False Positives spiked to 6,967, causing our Precision to drop to 0.17. Our F1-Score slightly improved to 0.51."

Slide 20: Final Model Weight BalancingScript:
"In our final attempt, we used Weight Balancing. This tells the algorithm to treat each 'True' case as more important than a 'False' case. The current results show a Recall of 0.67 and a Precision of 0.17.Our F1-Score rised by 0.01. Meaningless. After all these adjustments, the performance is still not 'high.' This led us to investigate if the problem wasn't the model, but the data itself."

Slide 21: The Big QuestionScript:
'Why so low performance?'. We had to test two hypotheses: Hypothesis 1: There exists a non-linear relationship that we are missing. Hypothesis 2: No relationship exists at all."

Slide 22: MIC (Maximal Information Coefficient)Script:
"To test this, we used the MIC, or Maximal Information Coefficient. Look at the scatter plots on the screen . Standard Pearson correlation, the correlation coefficient that we are used to, only works for straight lines. they can't detect the relations in the bottom row, but MIC can."

Slide 23: Visualizing MICScript:
"How does it work? MIC iteratively draws different grid over the distribution to find the one that best explains the data. Look at the 'High Information Cells' highlighted in green on the plots. By calculating how well the data partitions into these bins, MIC can catch any pattern, no matter how 'weird' or non-linear it is."

Slide 24: The Final Verdict on MICScript:
"Now, let’s read the final MIC results for our dataset. Look at the table. Every single row has an MIC score near zero. Statistically, any MIC below 0.1 means there is 'No Relationship'. The data is telling us clearly: these variables do not predict depression."

Slide 25: Hypothesis RejectionScript:
"We can now confidently reject Hypothesis 1 and confirm Hypothesis 2. the conclusion by so far, was that it is wrong to blame external circumstances—like how much a student studies or sleeps—as the sole cause of depression. Because our supervised models couldn't find a direct link, we decided to use Unsupervised Learning to see if there were hidden groups we were overlooking."
