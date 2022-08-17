Abstract:   According to BBC statistics, data changes for each earthquake that has occurred thus far. Thousands of people are killed, 50,000 are injured, 1-3 million are dislocated, and a considerable number of people go missing or become homeless. The structural damage is over 100 percent. It also has an impact on the economic loss, which ranges from ten to sixteen million dollars. A magnitude of 5 or higher is considered the worst. The most deadly earthquake to date happened in Indonesia, killing around 3 million people, injuring 1-2 million, and causing 100% structural damage. As a result, the effects of an earthquake are devastating, causing huge changes in everything from the environment and lifestyle to the economy. Each of these factors contributes to earthquake prediction. A precise forecaster is being constructed and developed in the current circumstances , a machine that will predict the future disaster. Using machine learning methods focuses on recognizing early indicators of an earthquake. The system is entitled to the fundamentals of constructing learning systems as well as the data science life cycle. Government sources provide data sets for the Indian subcontinent and the rest of the world. After preprocessing the data, a stacking model is built using RF and SVM algorithms. This mathematical model is created by algorithms using "training data-set".  The  models seeks out patterns which lead to disaster & adapts its structure to them in order to make decisions and forecasts without being explicitly adapted to carry out the mission. Keywords: Support vector Machine, Earthquake, Forecast, Machine Learning, Random Forest.
I.	INTRODUCTION : The link between earthquakes and structural damage and death continues to exist, and as a result, numerous areas, including seismology and environmental engineering, but not limited to these, are focused on it[1]. Its relevance extends to human existence as well, as it is required for survival and sustenance. A accurate and reliable prediction is essential for all disaster-prone areas, including those with few to no chances. It'll get us ready for the worst-case scenarios as well as the required steps that can be performed ahead of time to address any looming crises. The idea of saving lives is being studied with the help of excellent machine learning algorithms and data science to deliver accurate forecasts as technology advances and supports humans in living a better and more convenient existence. Artificial Intelligence includes Machine Learning. It enables the system to adapt to a given type of behaviour based on its own learning and to develop itself without the need for explicit programming, human intervention, or assistance[8]. Start by giving the algorithm a high-quality data set to work with (s). Algorithms find patterns and trends in data, as well as execute knowledge discovery and statistical analysis. Algorithm selection is based on data and the task that needs to be automated. Our goal is to improve our ability to predict and respond to catastrophic events. Excellent forecasts and warnings save lives. A warning of an impending disaster can be issued well in advance, which will aid in the reduction of both death and structural damage. Regression and classification models are the two types of predictive models created by ML algorithms[6]. Each one takes a different approach to data. The system in question uses a regression model, whose main goal is to predict a numerical result.
I.	Earthquake Forecast
           It is considered impossible to predict a seismic occurrence. It's a difficult task because of the event's nonlinearity and unreliability [3], Machine learning algorithms' ability to generate prediction models has transformed it into a potential marvel.
           The utilisation of their earthquake catalogue, or data-set, is required for earthquake prediction in the Indian subcontinent and the rest of the world. A complete list of past earthquakes, including their location, timing, magnitude, and depth, is referred to as an earthquake catalogue[3]. The process is based on defining adequate, necessary and appropriate factors, discovering patterns in these characteristics, and recognizing historical earthquake correlations in order to forecast future occurrences.
         A variety of RF, SVM Ensemble models are studied, modelled, and implemented. 

II.	RELATED WORK
                Following a major earthquake, Wenrui Li, Nakshatra, Nishita Narvekar, Nitisha Raut, Birsen Sirkeci, and Jerry Gao establish the concept of aftershocks. P-wave and S-wave arrival times can be utilised to locate the site of these aftershocks. To study patterns in P- and S-waves, the authors used data obtained in the SAC file format, which integrates data in a time series and is a waveform, from 16 seismic stations By using a triggering algorithm and filters, data is clipped and noise is removed, leaving only the required waveform. The p-wave and s-wave arrival times were calculated using the AR picker algorithm and then used as extracted features. Following that, the waveform is transformed to ASCII.
             Data is loaded into SVM, Decision Trees, RF & Linear Regression for comparison. RF is the most accurate in distinguishing leading data from earthquakes and non-earthquakes a 90% accuracy rate. Using the triangulation technique, determine the epicentre, forecast P-wave and S-wave arrival timings, and the difference between them. [2]
 
             Khawaja Muhammad Asim, Adnan Idris, Francisco Mart'nez-A' lvarez, and Talat Iqbal forecast earthquakes for the Hindu-Kush region using Random forest, rotation forest, and rotboost are three tree-based ensemble classifiers. They use the binary classification notion to an earthquake data collection and convert from binary to magnitude classifications. A novel feature set based on three factors: The Gutenberg-Richter relationship, seismic rate changes, and foreshock frequency distribution are all factors to consider. The estimation of 51 seismic features utilising appropriate methodologies and methodology is a standout aspect. We may infer that the method of computing 51 features was really effective because all of the models performed exceptionally well. The rotation forest model has a precision of 95.9%, making it the best of the rest models The useful insights for us come from the fact that while a prediction model must be created for every location on the planet, there is no way of predicting when or to what magnitude an earthquake will strike. To construct a categorization model, G.T. Prasanna Kumari use ensemble learning methods. The emphasis is on two well-known ensemble algorithms, Bagging and Boosting, to examine how the formation of varied ensembles enhances algorithm precision and how they differ in efficacy from the traditional strategy of generating a single model, which is often employed in ML to generate classifiers. Bagging and Boosting are thoroughly discussed, with information on how each algorithm's process flow differs from the other, as well as different applications, methodologies, power, accomplishments, and limitations. She goes on to discuss the differences between batch processing (all data supplied at once) and online processing (data generation in a continuous manner). 

While ensembles are often thought to be impractical for systems that perform online processing, she claims that they outperform batch processing and take less time to operate, especially for larger data sets. [4] Her advice will assist us in developing our own ensemble models.
        Ant'onioE Ruano, Maria G. Ruano, Pedro M. Ferreira, Ozias Barros, G.Madureira, and Hamid R.Khosravani obtain seismic data from the seismic monitoring system's PVAQ and PESTR stations. They point out an important objective fact: existing detectors at such stations generate a large number of false alarms and fail, because they are based on a standard STA/LTA ratio, they can identify the event. As a result, they present a novel new seismic detector known as the SVM classifier, whose use on such stations is unstoppable. They assess specificity and remember the measurements collected for each station, concluding that the SVM classifier may be able to successfully distinguish between noise and seismic activity. Following that, they concentrate their efforts on reducing the detection time in the Early Warning System. Because the obtained outcomes (88 and 110 seconds) are too large to be considered for deployment, a new approach with overlapping windows is inherited, and the time acquired is now 1.three seconds and 1.eight seconds, respectively. On the other side, an increase in do not forget and specificity levels leads to an increase in correct detection as well as phoney alerts. [5]



III.	SYSTEM DESIGN
        It takes time to develop predictive modelling. Python, Hadoop, and R are some of the most commonly used tools for model development.
The following are some of the steps that must be completed:
A.	DATA ACQUISITION
                       The process of bringing data into the system for production use, either from outside sources or from data created by the system, is known as data acquisition. This is the foundational step to take, and it refers to acquiring the necessary data. The essential data sets can be found on government websites, such as –
• The United States Geological Survey (USGS.gov) is a federal scientific organisation in the United States.
[13]
• The India Meteorological Department (IMD.gov) is an Indian government agency under the Ministry of Earth Sciences.
[14]
Google has purchased Kaggle houses a variety of data sets gathered from various government entities.
The data-set has the following columns:
•	Date
•	Time
•	Latitude
•	Longitude
Basic data into a tidy data package that may be used. It consists of two steps:
• Feature Development
• Data Science and Engineering
Data Engineering
              Real-World Data is not in a structured or compatible format, and part of it may be wrong, data that is incorrect, out of range, off-base, impossible, or lacking have an impact on the conclusions and make them dishonest, deceptive, and incorrect. In the training phase, irrelevant and erroneous information may make pattern identification and knowledge discovery even more challenging. As a result, it is the most important innovation in an ML framework, and data cleaning is required to remove or validate/correct such qualities. It entails data integration, missing value computation, categorical value handling, Error repair and transformation
Feature Engineering
                Selection of features, extraction of features, and scaling of features are all steps in the feature extraction process. A collection of data , may have a large number of random features that are useless for prediction. Feature Engineering is the process of taking a set of random features and reducing them to a set of essential characteristics , that aid in effective forecasting. For feature selection/extraction, ML provides a number of algorithms. A approach for standardising or normalising the scope of characteristics In a data collection, feature selection is important. The term "feature engineering" refers to the process of creating new features is advantageous because it reduces the size of information., decreases area for storing, speeds up calculation, and eliminates superfluous features.
B. MODEL BUILDING
       A  'model' is the outcome of a machine learning algorithm. First, the target and feature variables are deciphered and retrieved. Data is divided into two categories: training and testing and the training data is used to build and fit the regressor/classifier model.
CONFUSION MATRIX
POSITIVE        PREDICTED       NEGATIVE

TRUE POSITIVE
(TP)
CORRECT OBJECTS	
FALSE NEGATIVE
(FN)
MISSED OBJECTS

FALSE POSITIVE
(FP)
EXTRA OBJECTS	
TRUE NEGATIVE
(TN)
NO OBJECTS
           Precision              Negative predictive Values
           TP∕(TP+FP)                     TN∕(TN+TP)

Recall= TP∕(TP+FN)
Acurracy= (TP+TN)∕(TP+TN+FP+FN)
Precision= TP/TP+FP
Specificity= TN ∕ (TN+FP)
F1= 2*Recall*precision ∕ recall+precision

Scikit-learn is a Python open source package that implements a variety of ml methods, including an Through categorization, regression, and clustering. [15] The following is the construction procedure:
Building A Random Forest Regression Model :
            RF are  a type of learning in groups technique may used as well as classification and regression tasks. During training, It accepts the challenge  duty creating many decision trees and class outputs each individual tree's mean prediction (regression) or the classes' mode (classification) . A forest is represented by this massive amount of trees. The decision tree algorithm will generate rules for you to follow conduct out classification on a given 
targets and features are included in the training data set.
    • Magnitude
    • Depth
C. DATA PRE-PROCESSING
             The term "data pre-processing" refers to a process for converting input into output. Nodes will be features, and their presence or absence will indicate likelihood. This aids in the creation of a set of guidelines to follow. gini index[9] is used to determine the root and splitting nodes. The root and dividing nodes in Random Forest are calculated at random[9].

 
Fig. 1. Random Forest

As a result, a random-forest is model made up of multiple trees that can make decisions based on rules, with the randomly chosen root and parent nodes.

Building A Support Vector Machine Regression Model:
SVM, a supervised learning method, can perform regression and classification tasks. SVM uses a decision line called the hyperplane to separate distinct data classes. SVR tries at a specific departure from €, which is a threshold value for everyone, find a decision border for the function f(x) predictions to exist from within  the Yi, the initial hyperplane, was obtained as a target value, with data point falling within the boundary line. when predicting a numerical value. A judgement boundary that admits errors within a specified range is known as the Margin of Tolerance. [10] [11][12]


Fig. 2. SV Regressor

Building A Stacking Regressor Model :

             Bagging, Boosting, and stacking are just a few of the strategies to ensemble models in machine learning. Stacking is a prominent ensemble machine learning strategy for predicting several nodes and improving performance of the model. It allows to train numerous model to address the similar issues & then create a new model with superior performance based on their combined output.
A ml model that integrates the predictions of two or more models is known as an ensembled model.
A method of ensemble learning is stacking regression. As a result of the collaboration of  Using the meta-features generated by independent regression models using an an absolute training set as a basis, a meta-regressor is developed that determines the best fit. [7] Accuracy is often utilised. Our model is depicted in Fig 3. “R1” & “R2” RF and SVR, on the other hand, are two different types of regression models.
 
Fig. 3. Stacking

ARCHITECTURE OF STACKING:
          The stacking model's architecture is built around a minimum of two base/learners models, as well as a metamodel combines the base models' prediction. Level 0 models are the foundations, while level 1 models are the meta-models. Original (training) data, first-level models, first-level predictions, second-level models, and final predictions are all included in the stacking ensemble approach.

1.	The most popular way to create training datasets for meta-models is to divide them into n-folds using the RepeatedStratifiedKFold.
2.	The first fold, which is n-1, is now fitted to the base model, and it'll tell you what the nth folds will be.
3.	x1 train list is updated with the prediction given in the previous phase.
4.	Steps 2 and 3 should be repeated for the remaining n-1folds, resulting in an ‘x1’ n-dimensional training array
5.	the models has been all n areas of the body have been trained, it can make prediction about the data from the sample.
6.	This prediction will be added to the y1 test list.
7.	Similarly, utilising Model 2 and 3 for training, we may discover ‘x2’ train, ‘y2’ test,  ‘x3’ train, and ‘ y3’ test to obtain Level 2 forecasts
8.	The Meta model should now be trained. using Predictions at the first level are treated as features.
9.       Finally, in stacking Meta-learners can now use this model anticipate test data.
 
Fig.4. Architecture for stacking
Fuzzy K-Means Clustering:
             The Algorithm behind FKM is identical to K-means, a popular simple clustering technique. The sole difference is that instead of allocating a point to one cluster exclusively, it can have some fuzziness or overlap between two or more clusters.
The following are the main features of Fuzzy K-Means:
• Unlike K-Means, which looks for hard as a foundation a complete training set FKM looks for overlapping softer clusters.
• A single point in a soft cluster can belong to multiple clusters, each with its own affinity value, which is proportional to the distance from the cluster centroid. 
• Like KM, Fuzzy K-Means operates on objects with a defined distance measure that may be expressed in an n-dimensional vector space.
The following are the critical parameters for Fuzzy K-Means implementation:
• For the input, you'll need a Vector data set.
• The initial k clusters must be seeded with the RandomSeedGenerator.
• A Squared Euclidean Distance Measurement is required for distance measurement.
• If the squared value of the distance measure was utilised, a large convergence threshold, such as –cd 1.0, was chosen.
• A number representing the maximum number of iterations; the default is -x 10.
• A value of larger than -m 1.0 for the normalisation coefficient or fuzziness factor.
Deep Belief  Network:
How did Deep Belief Neural Networks Evolve?
              Perceptrons are used in the First Generation of neural networks to recognize a certain object or anything else by weighing it. On the other hand, perceptrons may be useful for simple technology but not for advanced technology. To address these concerns, back propagation was introduced to the Second Generation of Neural Networks, which compares the The received output is converted to the desired output, and the error value is reduced to zero. Then there were belief networks, which were directed acyclic graphs that were useful for inference and learning problems. Then we'll use Deep Belief Networks to help us generate unbiased values that we can store in leaf nodes.
Restricted Boltzmann Machines
RBMs are a type artificial neural network with generative stochasticity uses its inputs to learn a probability distribution. RBM is also used by deep learning networks. DBN, can be created, in example, by stacking RBMs and fine-tuning the resulting deep network using gradient descent RBMs are stacked and the deep network that results is fine-tuned and back propagation.

The Architecture of DBN
 
             A DBN is created by connecting a sequence of constrained Boltzmann machines in a certain order. The outcome of the Boltzmann machine's "output" layer is supplemented as input to the next Boltzmann machine in a sequential manner. Then we'll train it till it converges, and then we'll apply it until the entire network is completed.
The operational pipeline of Deep Belief Network looks like this:
• To pre-train DBN, we'll utilise the Greedy algorithm. The greedy learning method, which uses a layer-by-layer approach, is used to learn top-down generating weights. The relationship between variables in one layer and variables in the next layer is determined by these generative weights.
• On the top two hidden layers of DBN, we perform many steps of Gibbs sampling. The RBM is defined by the As a result, this stage effectively pulls a sample from the top two buried layers.
• Then, using Create a sample from the visible units by doing an ancestral sampling pass across the rest of the model.

Fuzzy C-Means Clustering:
               Clustering is a technique for unsupervised machine learning for dividing a population into multiple groups. groups or clusters, the same group's data points are similar, but data points from separate groups are different.
• The data points in a cluster are close to one another, making them very similar.
• Input data points in separate clusters are spread out and distinct from one another.
Clustering is a technique for identifying segments or groupings in a dataset. The Fuzzy C-means clustering (FCM) Algorithm is a popular soft clustering algorithm.
Clustering with fuzzy C-Means is a soft clustering technique approach in which each data point is assigned a probability or likelihood score for belonging to that cluster.
• Fix the value of c (number of clusters), select a value of m (usually 1.25m2), and initialise partition matrix U.
• Determine the cluster centres (centroid).
• Revise the Partition Matrix
• Repeat steps 1-3 until convergence is achieved.
Convolution Neural Networks(CNN):
            Convolution Computational neural networks, often known as covnets, are neural networks with shared parameters. Consider the following scenario. It can be visualised as a cuboid with length, width, and height (as images generally have red, green, and blue channels). Now picture Applying a small neural network to a small area of this image, say with k outputs, and then vertically showing the results. We can get a new image with varying width, height, and depth if we move the neural network throughout the entire image. Although they are shorter and thinner, there are now more than simply R, G, and B channels. Convolution is the name for this procedure. A normal neural network will have a patch size equal to the image size. There are less weights because of this small section.
ConvNet layering:
            A covnet is a collection of layers, each of which transforms one volume into another using a differentiable function.
Layers can be divided into several categories.
Let's look at an image with the dimensions 32 x 32 x 3 and run a covnets on it.
1.	Input Layer: The image's raw input is stored in this layer, this is 32 pixels wide, 32 pixels tall, and three pixels deep.
2.	Convolution Layer: This layer's All filters and picture patches are added together to form a dot product. It's utilised to figure out how much production you're going to get. The output volume of this layer will be 32 x 32 x 12 if there are 12 filters overall.
3.	Activation Function Layer: The output of the convolution layer will be activated layer by layer by this layer. Frequent activation functions include Others include RELU: max(0, x), Sigmoid: 1/(1+e-x), Tanh, and Leaky RELU. Because the volume remains the same, the 32 x 32 x 12 will be the output volume.
4.	The Pool Layer: is injected on a regular basis, into the covnets, and primary function is to reduce the volume size, which reduces computing time, saves memory, and avoids overfitting. Maximum pooling and average pooling are two popular types of pooling layers. The final volume will be 16x16x12 if we use a maximum pool with 2 x 2 filters and stride 2.
5.	Fully-Connected Layer: This is a simple neural network layer that takes data from the previous layer, computes class scores, and produces a one-dimensional array with the same number of classes as the previous layer.

Long Short Term Memory (LSTM):
             Long short-term memory is a sort of recurrent neural network. The previous step's output is used as the input for the subsequent RNN phase. The LSTM was designed by Hochreiter & Schmidhuber. It addressed the problem of RNN long-term reliance, which occurs when produce more accurate predictions when using current data. As the gap length increases, RNN does not deliver an efficient performance. LSTM may store data for a long time by default. It's used to analyse, predict, and classify time-series data. The LSTM is made up of four neural networks and discrete Cells are a type of memory block. The cells hold data, and the gates control memory.

Recurrent Neural Networks (RNN):
           RNNs are a type of neural network that is both powerful and resilient, and they are one of the most promising algorithms currently available. Because these are the only ones with internal memory, they are used.
Many additional deep learning techniques, such as recurrent neural networks, are relatively recent. In the 1980s, they were first developed but we have only just tapped into their full potential RNNs became popular in the 1990s as a result of greater computer power, large amounts of data, and the introduction of LSTM. RNNs are a type of neural network that can be used to model a sequence of data. RNNs, which are derived from feedforward networks, are a type of artificial intelligence share the same characteristics as human brains. Simply said, recurrent neural networks are better at anticipating sequential data than other algorithms.

How Recurrent Neural Networks Work?
            To fully comprehend RNNs, you'll need to be familiar with sequential data, which is just a collection of data points ordered in chronological order.
             Sequential data is simply sorted data in which related items appear in a sequential manner. Financial data and DNA sequences are two examples. Time series data, which is simply a collection of data points, is probably the most popular sort of sequential data.
RNNS ARE OF THE FOLLOWING TYPES:
• One to One
• One to Many
• Many to One
• Many to Many
             
RNNs can map one to many, many to many (translation), and many to one, whereas Only one input can be translated into one output by feed-forward neural networks. temporally (classifying a speech).

D. PREDICTIONS Algorithm:
1. Load libraries and input data sets.
2. Pre-processing of data
3. Model Construction
4. Prediction Making



IV.	RESULT

              For large datasets, the combination of the random forest and support vector machine models works well. In comparison to the bagging and enhancing precision, the accuracy of the The stacking model has the highest score of 83.percent. For each methodology, the response time is the same. The time spent training for stacking is slightly longer.
 
 
Fig.5. Represent the direction of earthquake
Table: shows the primary, secondary wave, direction and the estimated value of earthquake in kilometres.
p-wave 	p-wave	s-wave in EW component	s-wave in NS component	Estimated value
72.31	66.32	72.58	72.31	0.04
662.55	654.35	725.80	723.15	484.80
72.31	66.33	72.58	72.31	0.04
68.85	63.18	69.12	68.87	0.15
72.13	66.32	72.58	72.31	0.04
69.54	63.79	69.78	69.53	-0.11
72.31	66.33	72.58	72.315	0.04
71.55	69.00	71.86	71.59	0.35
62.29	57.15	62.03	61.80	-3.93


 



V.	CONCLUSION
         Machine learning technology combined with seismic activity produces efficient and effective results.  As a result, we can infer that combining meaningful results that may be used to anticipate earthquakes in a wide range of situations, provided that the prior history of the event is preserved. Our effort was fruitful. The two can work together much more effectively to better protect against earthquakes. The importance of large datasets is demonstrated. Area-centric prediction models can be applied, boosting the probability of accurate prediction tenfold. However, this comes at the expense of researching the methods used to construct the Stacking model, as the metaregressor will only perform successfully if the algorithms used to construct it are accurate. The methodology's use could be expanded to include the prediction of numerous natural disasters.



VI.	REFERNCES

1.	W. Ma, P. Dang, Z. Xie and J. Lu, "Application of stochastic finite-fault method to marine earthquake : 2021 Fukushima earthquake in Japan," 2021 7th International Conference on Hydraulic and Civil Engineering & Smart Water Conservancy and Intelligent Disaster Reduction Forum (ICHCE & SWIDR), Nanjing, China, 2021, pp. 976-980.
2.	R. M. Labanan and R. C. Raga, "A Study of Macroseismic Intensity Prediction using Regression of Philippine Earthquakes," 2021 1st International Conference in Information and Computing Research (iCORE), Manila, Philippines, 2021, pp. 129-134.
3.	M. Hu, Q. Liu and H. Ai, "Research on Emergency Material Demand Prediction Model Based on Improved Case-Based Reasoning and Neural Network," 2021 IEEE 3rd International Conference on Civil Aviation Safety and Information Technology (ICCASIT), Changsha, China, 2021, pp. 270-276.
4.	M. Hu, Q. Liu and H. Ai, "Research on Emergency Material Demand Prediction Model Based on Improved Case-Based Reasoning and Neural Network," 2021 IEEE 3rd International Conference on Civil Aviation Safety and Information Technology (ICCASIT), Changsha, China, 2021, pp. 270-276.
5.	O. M. Saad et al., "Machine Learning for Fast and Reliable Source-Location Estimation in Earthquake Early Warning," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022.
6.	Y. Garg, A. Masih and U. Sharma, "Predicting Bridge Damage During Earthquake Using Machine Learning Algorithms," 2021 11th International Conference on Cloud Computing, Data Science & Engineering (Confluence), Noida, India, 2021, pp. 725-728.
7.	R. Kail, E. Burnaev and A. Zaytsev, "Recurrent Convolutional Neural Networks Help to Predict Location of Earthquakes," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 8019005.
8.	S. Sarkar, A. Roy, S. Kumar and B. Das, "Seismic Intensity Estimation Using Multilayer Perceptron for Onsite Earthquake Early Warning," in IEEE Sensors Journal, vol. 22, no. 3, pp. 2553-2563, 1 Feb.1,2022.

9.	A. Wibowo et al., "Earthquake Early Warning System Using Ncheck and Hard-Shared Orthogonal Multitarget Regression on Deep Learning," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022.
10.	L. Yang, Z. Shan, R. Ma and L. Jing, "Influence of the Warning Area Division on the Effect of the Propagation of Local Undamped Motion (PLUM) Method," 2021 7th International Conference on Hydraulic and Civil Engineering & Smart Water Conservancy and Intelligent Disaster Reduction Forum (ICHCE & SWIDR), Nanjing, China, 2021, pp. 32-36.
11.	B. Feng and G. C. Fox, "Spatiotemporal Pattern Mining for Nowcasting Extreme Earthquakes in Southern California," 2021 IEEE 17th International Conference on eScience (eScience), Innsbruck, Austria, 2021, pp. 99-107.
12.	N. Kato, "Propagation of a precursory detachment front along a seismogenic plate interface in a rate–state friction model of earthquake cycles," in Geophysical Journal International, vol. 228, no. 1, pp. 17-38, Aug. 2021.
13.	Y. Fan and Z. Chen, "Research on Abnormal Geomagnetism Prediction Method based on Electromagnetic Spectrum I/Q Signal Detection," 2021 5th Asian Conference on Artificial Intelligence Technology (ACAIT), Haikou, China, 2021, pp. 348-352.
14.	Z. Yu, K. Zhu, K. Hattori, C. Chi, M. Fan and X. He, "Borehole Strain Observations Based on a State-Space Model and ApNe Analysis Associated With the 2013 Lushan Earthquake," in IEEE Access, vol. 9, pp. 12167-12179, 2021.
15.	J. Min, B. Ku and H. Ko, "Feedback Network With Curriculum Learning for Earthquake Event Classification," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 7504505.
16.	D. Jozinović, A. Lomax, I. Štajduhar and A. Michelini, "Transfer learning: improving neural network based prediction of earthquake ground shaking for an area with insufficient training data," in Geophysical Journal International, vol. 229, no. 1, pp. 704-718, Oct. 2021.
17.	E. Kiser and H. Kehoe, "The hazard of coseismic 
