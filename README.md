# cs482-buildings-project
Provided in this repo is a dockerfile for creating and running an environment for this, a tutorial like notebook for you to see the creation and testing of a pipeline to try and detect buildings, the dataset we trained and tested on, as well as the scripts we used in addition to the notebook.

When running this project, it may be nessecary to change around path names in many places so it is something you should keep note of. 

[Here]("https://drive.google.com/drive/folders/1xgG7AYD0U09MxRo_UEhGz5fT7hd9nZ8L?usp=sharing") is the google drive link for our prototyping step in colab if you want to look at that as well as the dataset that we used of buildings in Paris. There is also the generate masks script as well as a copy of the github used for a unet network. However as mentioned before there were a lot of path changes needed to use the unet network as well as some changes in the code to use our dataset. For example, the dataset that the creator of the unet model used had 5 classes whereas we only had 1. Image sizes were different as well as tif depth. So if you are trying to use your own dataset, some of these things may need to be changed around.

