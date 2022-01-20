# capstone-Azure-MachineLearning

# Overview 
This project is part of the Udacity's Azure ML Nanodegree. In this project, we were asked to use a dataset of our choice to solve a Machine Learning problem using Azure ML. To do so, we need to train models using AutoML as well as Hyperdrive, then we choose the best model, deploy it and consume it.
# DataSet
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Column	Description
age	Age of the patient
anemia	Decrease of red blood cells or hemoglobin (boolean)
creatinine_phosphokinase	Level of the CPK enzyme in the blood (mcg/L)
diabetes	If the patient has diabetes (boolean)
ejection_fraction	Percentage of blood leaving the heart at each contraction (percentage)
high_blood_pressure	If the patient has hypertension (boolean)
platelets	Platelets in the blood (kiloplatelets/mL)
serum_creatinine	Level of serum creatinine in the blood (mg/dL)
serum_sodium	Level of serum sodium in the blood (mEq/L)
sex	Woman or man (binary)
smoking	If the patient smokes or not (boolean)
time	Follow-up period (days)
DEATH_EVENT	If the patient deceased during the follow-up period (boolean)


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

 # Model Deployment
  Based on the previous results, I chose the Voting Ensemble model as it has the best Accuracy out of the two. To successfully deploy the model, we must have an InferenceConfig and an ACI Config.
# Register the Model
description = 'AutoML Model trained on heart failure data to predict if death event occurs or not'
tags = None
model = remote_run.register_model(model_name = model_name, description = description, tags = tags)
# Define an Entry Script
The entry script receives data submitted to a deployed web service and passes it to the model. It then takes the response returned by the model and returns that to the client. For an AutoML model this script can be downloaded from files generated by the AutoML run. The following code snippet shows that.
script_file_name = 'inference/score.py'
best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'inference/score.py')
# Define an Inference Configuration
An inference configuration describes how to set up the web-service containing your model. It's used later, when you deploy the model.
inference_config = InferenceConfig(entry_script=script_file_name)
# Define a Deployment Configuration
aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "hfData", 'type': "automl_classification"}, 
                                               description = 'Heart Failure Prediction')
# Deploy the Model
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)

![image](https://user-images.githubusercontent.com/59172649/150383500-e8283a1d-304b-4b79-8e59-932a7b3251a0.png)
Once the model is deployed the model endpoint can be accessed from the Endpoints sections in the Assets Tab
![image](https://user-images.githubusercontent.com/59172649/150383743-ce8ea538-0321-4394-a507-122da73bb7b6.png)
The deployment state of the model can be seen as Healthy which indicates that the service is healthy and the endpoint is available.
![image](https://user-images.githubusercontent.com/59172649/150383801-99898218-6eb1-405e-95c7-0d54beb832a3.png)
![image](https://user-images.githubusercontent.com/59172649/150383821-6eac954c-ca82-465c-8b6b-3a37ce9caeb4.png)
Once the model has been deployed, requests were sent to the model. For sending requests to the model the scoring uri as well as the primary key (if authentication is enabled) are required. A post request is created and the format of the data that is needed to be sent can be inferred from the swagger documentation:
![image](https://user-images.githubusercontent.com/59172649/150384069-16da5b19-37fd-4bd9-bc50-2802ad397877.png)
The following code interacts with the deployed model by sending it 2 data points specified here and in the data.json file.
import json

#URL for the web service, should be similar to:
#'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = aci_service.scoring_uri
#If the service is authenticated, set the key or token

#Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 70.0,
            "anaemia": 1,
            "creatinine_phosphokinase": 4020,
            "diabetes": 1,
            "ejection_fraction": 32,
            "high_blood_pressure": 1,
            "platelets": 234558.23,
            "serum_creatinine": 1.4,
            "serum_sodium": 125,
            "sex": 0,
            "smoking": 1,
            "time": 12
          },
          {
            "age": 75.0,
            "anaemia": 0,
            "creatinine_phosphokinase": 4221,
            "diabetes": 0,
            "ejection_fraction": 22,
            "high_blood_pressure": 0,
            "platelets": 404567.23,
            "serum_creatinine": 1.1,
            "serum_sodium": 115,
            "sex": 1,
            "smoking": 0,
            "time": 7
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

#Set the content type
headers = {'Content-Type': 'application/json'}
#If authentication is enabled, set the authorization header

#Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
** The result obtained from the deployed service is- **
![image](https://user-images.githubusercontent.com/59172649/150384220-ed67f2ed-4456-47b4-b083-1316236cc0a8.png)

 The requests being sent to the model can be monitored through the Application Insights URL (If Application Insights are enabled) along with failed requests, time taken per request as well as the availability of the deployed service.
 ![image](https://user-images.githubusercontent.com/59172649/148778479-ca1cea18-d75b-4485-8bfa-61e0939611df.png)


# Screen Recording 
https://drive.google.com/file/d/1qs9fUU2j-yirxYUmWRnLeEUWEhO4FMfS/view?usp=sharing
# Standout Suggestions 
To improve this project in future, I can make the following improvements:

    -Choose another primary metric like "AUC Weighted" or F1 Score.
    -Choose another Classifier instead of Logistic Regression.
    -Use another dataset with more entries.
    -Train the model longer.
    -Choose another sampling policy.














