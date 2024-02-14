# Training with AIchor the IG framework


## AIchor Introduction
If you want to simply experiment with your Machine Learning code using Instadeep’s best-in-class hardware and devops practices, this page is the place to start.

AIchor is Instadeep in-house ML DevOps system.

AIchor allows you to plug your Git source code repository and run ML experiments.
All experiments are triggered by a Git commit.
You can always trace back the generated files and results to the state of your code at that specific commit.
Several tools are handily available in the Ichor tool to monitor you experiment: logs, resources, tensorboard, and more to come.

If you want to know more about AIchor, you can visit the [AICHOR user manual notion page](https://www.notion.so/instadeep/User-Guide-6af905b17f734281a944d90084b9de2c)



## How to launch an experiment with AIchor

### Move Manifest file

Before launching the experiment, please make sure to move the manifest.yaml file to the root folder.
This way AIcor will be able to catch the experiment.

### Update files

#### runAIchor.sh

First put the datasets (train/test) under the datasets directory in the AIchor plateform.
Under the runAIchor.sh file update the following based on your experiment:
- test_path='path to the train csv'
- train_path='path to the test csv'

All the datasets can be accessed in the training from the PVC: '/mnt/dataset/'
Note: Do not forget to push the features.yml file in the input data

Under the runIchor.sh file you need to update the train, test, configuration file and folder name of the experiment.

the 'out' folder contains the results of the experiment, you can check it using the AIchor UI plateform.


#### Manifest file

A manifest file should be always under the root directory / to launch the experiments:
You can change the image name and it uses the Dockerfile.Aichor to build the docker image on AIchor.
The command runned will be the bash script, you can change it accordingly.

By default the number of CPUs is 50 but you are free to use more.

## Run using the CI from your terminal

To launch an experiement:
* stage the modified files
* run the commit command with EXP key

```bash
git commit -m 'EXP: test training IG framework with AIchor'
```
if you did not change anything you can run the following:

```bash
git commit -m 'EXP: test training IG framework with AIchor' --allow-empty
```

* push changes to gitlab


## Output folder

The output folder can be found on AIchor under the datasets directory/outputs/output/


## Multi train distributed

To run the multi train command in distributed mode on different worked:

* In the manifest file, you need to change the worker count paramter depanding on how many experiements you have.
-->  For exmaple, the muli train configuration command has 4 experiments (bnt, bnt_biondeep, bnt_biondeep_pmhc, bnt_pmhc)
we therefore change the number of workers to 4.

* Change the bash script command from running the default AIchor bash to running the runAIchor_multi_train.sh file.

* Make sure to change the train and test paths in the .sh file for multi training.


🚚 **Upload data folder**

Using the AIchor interface: navigate to the Datasets tab → Inputs → Create Folder → Choose folder name, e.g. my_data → Upload File.

```bash
# Data directory
gsutil cp -r my_data gs://biondeep-ig-28558948a00b44a8-inputs/

# File(s) in Data directory
gsutil cp my_data/train.csv gs://biondeep-ig-28558948a00b44a8-inputs/

```

🚀  **Triggering the pipeline**

In order to trigger the pipeline, you have to push a commit to the **biondeep-ig** gitlab repo.

```bash
git add -u # add current changes to stage
git commit -m "EXP: multi train IG framework with AIchor" # commit, starting with 'exp: ' so that AIchor triggers the run
git push --set-upstream origin <branch name>
```

if you did not change anything you can run the following:

```bash
git commit -m 'EXP: multi train IG framework with AIchor' --allow-empty
git push
```
📤  **Output**

When the job is complete, you will be find the outputs on the output bucket, note that the folder will have a name which matches the hash for the experiment, for example in the image above we can read off the experiment hash as 5bfc0987-45dc-4b21-80b7-8a47c2e471f8.

To download the output files, use the gsutil cli, for example:

```bash
gsutil cp -r gs://biondeep-ig-28558948a00b44a8-outputs/output/5bfc0987-45dc-4b21-80b7-8a47c2e471f8/ models/
```

📃 **Logs**

The complete logs, i.e. including debug data, can be downloaded from the AIchor's bucket:

```bash
gsutil cp -r s3://biondeep-ig-28558948a00b44a8-outputs/logs/5bfc0987-45dc-4b21-80b7-8a47c2e471f8/ .
```

## Important note:

By default the dockerignore will ignore all by default to reduce build context as much as possible.
You can find that both

```
!runAIchor.sh
!configuration
```

were removed from the ingore to run the model with.
