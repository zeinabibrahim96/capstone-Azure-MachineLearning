# capstone-Azure-MachineLearning
# Overview 
This project is part of the Udacity's Azure ML Nanodegree. In this project, we were asked to use a dataset of our choice to solve a Machine Learning problem using Azure ML. To do so, we need to train models using AutoML as well as Hyperdrive, then we choose the best model, deploy it and consume it.
# DataSet
For this project, I used the Heart Failure dataset from Kaggle.
![image](https://user-images.githubusercontent.com/59172649/148777619-e38b9caf-ebcb-4b20-bdfc-1b38d49afc6b.png)

# The Challenge 
The problem is to predict the value of the DEATH_EVENT target column. This is a classification problem (1: death, 0: no death).
# Access 
In Azure ML Studio, I registered the dataset from local files. I have the .csv file in my github repository and I downloaded it in the VM. For the train.py file I used the link to my repo to create a Tabular Dataset.
# Auto ML 
Regarding the Compute Target, I used a 'STANDARD_D2_V2' vm_size with max_nodes=4. For the AutoML Configuration, I used the following settings :

    automl_settings = {
        "experiment_timeout_minutes": 15,
        "iterations": 40,
        "max_concurrent_iterations": 4,
        "n_cross_validations": 3,
        "primary_metric" : 'accuracy'
    }
    automl_config = AutoMLConfig(compute_target=compute_target,
                                 task = "classification",
                                 training_data=dataset,
                                 label_column_name="DEATH_EVENT",
                                 enable_early_stopping= True,
                                 debug_log = "automl_errors.log",
                                 **automl_settings
                                )
 the AutoML Config:

   - experiment_timeout_minutes: I chose 15 minutes as the maximum amount of time the experiment can takee before it terminates because I have a small dataset with only 299 entries.
   - max_concurrent_iterations: Represents the maximum number of iterations that would be executed in parallel. The default value is 1.
   - n_cross_validations: To avoid overfitting, we need to user cross validation.
   - primary_metric: Accuracy.
   - task: Classification, since we want to have a binary prediction (0 or 1).
# Result 
Azure AutoML tried different models such as : RandomForests, BoostedTrees, XGBoost, LightGBM, SGDClassifier, VotingEnsemble, etc.

The best model was a Voting Ensemble that has Accuracy=0.86619

Ensemble learning improves machine learning results and predictive performance by combining multiple models as opposed to using single models. The Voting Ensemble model predicts based on the weighted average of predicted class probabilities.
![image](https://user-images.githubusercontent.com/59172649/148777728-9777f63f-4079-4512-9953-7659bab119a4.png)
![image](https://user-images.githubusercontent.com/59172649/148777760-0a90601e-1b6d-4e22-a78d-31e331c2217a.png)
![image](https://user-images.githubusercontent.com/59172649/148777803-b2b9eb89-3427-478f-869f-8dfc3624bb8c.png)
# Hyperparameter Tuning
For Hyperparameter Tuning, I used the Logistric Regression algorithm from the SKLearn framework. There are two hyperparamters for this experiment:

C | The inverse regularization strength. max_iter | The maximum iteration to converge for the SKLearn Logistic Regression.

I also used random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

The parameter search space used for C is [0.01, 0.1, 1.0, 10.0, 100.0] and for max_iter is [20, 50, 100, 120, 150]

The BanditPolicy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This helps to improves computational efficiency.
    # Create an early termination policy. This is not required if you are using Bayesian sampling.
    early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)


    # Create the different params that you will be using during training
    param_sampling = RandomParameterSampling(
        {
            '--C': choice(0.01, 0.1, 1.0, 10.0, 100.0),
            '--max_iter': choice(20, 50, 100, 120, 150)
        }
    )


    # Create your estimator and hyperdrive config
    estimator = SKLearn(source_directory='./', 
                        entry_script='train.py',
                        compute_target=compute_target)

    hdr_config = HyperDriveConfig(
        estimator=estimator, 
        hyperparameter_sampling=param_sampling, 
        policy=early_termination_policy, 
        primary_metric_name='Accuracy', 
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
        max_total_runs=20, 
        max_concurrent_runs=4
    )
  # The results 
  HyperDrive tested many combinations of C and max_iter and the highest accuracy that our Logistic Regression Model acheived was 0.788
  The hyperparamteres that were used by this model are:
  - Max Iterations (max_iter)=20
  -Regularization Strength (C)=100
  ![image](https://user-images.githubusercontent.com/59172649/148778274-d3616ba6-d53d-4b15-829a-f78c398ae5b0.png)
  ![image](https://user-images.githubusercontent.com/59172649/148778404-b6d6ae4c-b104-43a7-8466-c65327d1cc50.png)
![image](https://user-images.githubusercontent.com/59172649/148778421-aeef31b5-f0c5-41bb-b2a3-03ddbb2e1db5.png)
![image](https://user-images.githubusercontent.com/59172649/148778479-ca1cea18-d75b-4485-8bfa-61e0939611df.png)
 # Model Deployment
 Based on the previous results, I chose the Voting Ensemble model as it has the best Accuracy out of the two. To successfully deploy the model, we must have an InferenceConfig and an ACI Config.
 ![image](https://user-images.githubusercontent.com/59172649/148778987-5784812d-8a60-4a80-8b5f-66f7665d9fc6.png)
![image](https://user-images.githubusercontent.com/59172649/148779122-b4098e1f-265c-49a1-8bc1-59c68a884402.png)
# Standout Suggestions 
To improve this project in future, I can make the following improvements:

    -Choose another primary metric like "AUC Weighted" or F1 Score.
    -Choose another Classifier instead of Logistic Regression.
    -Use another dataset with more entries.
    -Train the model longer.
    -Choose another sampling policy.














Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
![image](https://user-images.githubusercontent.com/59172649/147488765-4f18be12-57db-470d-ae99-d66cfae939ca.png)
![image](https://user-images.githubusercontent.com/59172649/147488814-d4c7229a-e17d-4fd6-92d2-48fa18de5792.png)
![image](https://user-images.githubusercontent.com/59172649/147488896-1f66892f-d44f-4b72-b36e-eff3d365b3d6.png)
![image](https://user-images.githubusercontent.com/59172649/147488947-5ba7a457-db36-42c4-af38-444a29e7a55d.png)
![image](https://user-images.githubusercontent.com/59172649/147489092-20297ee7-bafd-4f04-88a3-863a89553162.png)
![image](https://user-images.githubusercontent.com/59172649/147747957-22475327-5c82-407a-bdd7-be8f2b6000eb.png)
![image](https://user-images.githubusercontent.com/59172649/147748033-a3fd12d8-67d1-4146-b43c-585c2c3c5f46.png)
![image](https://user-images.githubusercontent.com/59172649/147748540-3c03a02a-683c-45e8-abd6-e866fadf5f2a.png)
![image](https://user-images.githubusercontent.com/59172649/147749227-7e17a987-313b-4b8b-bf9d-0b8b2f92af6c.png)
![image](https://user-images.githubusercontent.com/59172649/147825962-930ef779-d38b-4e43-b32e-2a35b6aef89d.png)


