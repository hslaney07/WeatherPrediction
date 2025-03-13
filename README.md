# Weather Prediction using LSTM
**Built a Weather Prediction LSTM model. This model predicts the temperature high of a given day in Boston, MA using historical weather data.** 

Note: ChatGPT and DeepSeek were utilized to outline the initial code structure as well as help with wording in README, help create some graphs, and help with some bugs encountered.

### Dataset
- Used weather data from the **National Oceanic and Atmospheric Administration (NOAA)** for training and testing.
- The station used is **Boston Logan International airport in Boston, Massachusetts, US**
- **The initial training and testing data begins on 1/1/2020 and ends on 3/5/2025.**
- The notable data included in the CSV from NOAA includes: 
    - AWND (Average daily wind speed in mph)
    - PRCP (precipitation in inches)
    - SNOW (snowfall in inches)
    - TAVG (Average Temperature in Fahrenheit)
    - TMIN (Min Temperature in Fahrenhiet)
    - TMAX (Max Temperature in Fahrenheit)
    - Weather types. There are 9 different weather types, but only 8 are of note for this data.         
        - WT01 is fog, ice fog, or freezing fog (may include heavy fog). 
        - WT02 is heavy fog or heaving freezing fog (not always distinguished from fog). 
        - WT03 is thunder.
        - WT04 is ice pellets, sleet, snow pellets, or small hail
        - WT05 is hail
        - WT06 is glaze or rime
        - WT08 is smoke or haze
        - WT09 is blowing or drifting snow

### Long Short-Term Memory (LSTM)
- LSTMs are a type of Recurrent Neural Network (RNN) designed to handle long-term dependencies in sequential data. Unlike traditional RNNs, which struggle with retaining information over long sequences, LSTMs can remember important information for extended periods. This makes them particularly useful for time-series forceasting, such as weather prediction.
- After reviewing [PyTorch LSTM tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) and [GeeksForGeeks LSTM Tutorial](https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/), used ChatGPT to create the inital code structure including the LSTM model, preparing the data outline, and training outline.


**Implementation**
- The model can be found in **WeatherLSTM.py**
- The code used for training can be found in **train.py**
    - The only thing to note is the preparing data
        - We perform some data cleaning to prevent NaN errors
        - We also use MinMaxScaler to scale the data. This is a common practice for neural networks. We scale our data to be between -1 and 1.

### Experiments
**1. Initial Experimentation**
- The code for this experiment can be found in **LSTM_Initial_Experiment_config.py**
- In this design, there are several parameters that can be adjusted as well as aspects of the architecture design.
- There are 5 key parameters that were customized in the initial experiment:
    - **Input Data**: The data fed into the model could either be just the TMAX data or a combination of multiple data points (TMAX, AWND, TAVG, TMIN, WT01...WT09).
    - **Sequence Length**: This refers to the number of time steps in an input sequence that the LSTM processes at once. For example, if the sequence length is 10, the model uses the past 10 days of data to predict the temperature high for the next day. Sequence lengths ranged from 5 to 10 to 30.
    - **Hidden Size**: This represents the number of neurons or features in the hidden state of each LSTM cell. Hidden size values ranged from 50 to 100 to 150.
    - **Number of Layers**: This is the number of stacked layers in the LSTM model. The values varied from 2 to 6 to 10.
    - **Epochs**: This is the number of epochs used during training. Epoch values ranged from 100 to 200."
- In total, there are **108 different configurations explored**. For each experiment, the following details were recorded: the model configuration (as described in the previous section), the final training loss (MSE), the final validation loss (MSE), the test loss (MSE), the test loss (MAE), and information about the worst prediction. The worst prediction was analyzed to check for any overlap or patterns.
- To analyze performance, the top 5 configurations with the lowest test loss (MSE) were selected for further evaluation. Below are the details of these configurations:
![Top 5 Performing Stuctures](images_for_readme\Initial_Experiment_Top5_Results.png)

- Here is an image showing the performance of the best performing configuration. We are plotting the LSTM's predictions against the actual high temperature in Boston.
![Top Performing Stucture - Boston Temperature High vs Prediction](images_for_readme\alldata_sl=30_hs=150_nl=2_lr=0.001_e=200.png)

**2. Second Experimentation**
- Code for this can be found in **LSTM_Second_Experiment_config.py**
- Based on the above results, a second experiment began using the insight of the above structures to see if an even better structure could be found.
- Here are the parameter details for this experiment.
    - **Input Data**: The only data fed into the model was all of the data above (TMAX, AWND, TAVG, TMIN, WT01...WT09)
    - **Sequence Length**: Varies from 10 to 30 to 40
    - **Hidden Size**: Varies from 100 to 150
    - **Number of Layers**: Varies from 2 to 6 to 10.
    - **Epochs**: Varies from 200 to 250 to 300.
-  In total, there are **54 different configurations explored**. The same information was recorded as the initial experiment.
- To analyze performance, the top 5 configurations with the lowest test loss (MSE) were selected for further evaluation. Below are the details of these configurations:
![Top 5 Performing Stuctures](images_for_readme\Second_Experiment_Top5_Results.png)
- Here is an image showing the performance of the best performing configuration. We are plotting the LSTM's predictions against the actual high temperature in Boston.
![Top Performing Stucture 2 - Boston Temperature High vs Prediction](images_for_readme\alldata_sl=40_hs=150_nl=2_lr=0.001_e=200.png)


### Analysis of Experiments
**Analysis of the Initial Experiment**
- As noted, there were 108 different configurations explored. We determined the top 5 performing configurations by the test loss. It is important to note that the top 5 configurations also had the top 5 lowest training loss. 
- As can be seen the best performing configuration has a test loss of 3.87 and even the fifth best configuration has a test loss of 4.03 which are very low test loss values given that Mean Squared Error is used.
- Looking at the graph showing the acual temperature high vs the predicted temperature high in Boston, MA. We can see that the LSTM does well in general except for days with a large change in weather which is to be expected.
- **Findings**
    - The input data for the top 5 best configurations are all TMAX, AWND, TAVG, TMIN, WT01...WT09, and none are just TMAX. Thus, it seems that the extra information helps the model better predict the next day's temperature.
    - All of the top 5 best configurations have sequence length of 30 possibly indicating that higher sequences could result in better performance.
    - All of the top 5 best configurations have hidden sizes of 100 or 150, none are 50, suggesting that 100 or 150 could result in better performance.
    - All of the top 5 best configurations have 200 epochs which suggest that higher epochs could result in better performance. 


**Analysis of the Second Experiment**
- As noted, there were 54 different configurations explored. We determined the top 5 performing configurations by the test loss. In this case the top 5 best configurations did not also have the top 5 lowest training loss which could indicate other configurations were overfitting more to the training data.
- This experiment was specifically designed to find a better configuration given the findings above. 
- As can be seen the best performing configuration has a test loss of 3.84. When compared to the initial experiment, where the test loss was already relatively low, it's not surprising that the new configuration does not significantly outperform the previous one. Overall, the experiments were both able to find 5 well performing configurations with very little difference.
- The biggest difference in the experiments is that overall the average test loss of the initial experiment (48.89) is much larger than the average test loss of the second experiment (4.48) which is to be expected. 
- Looking at the graph showing the actual temperature high vs the predicted temperature high in Boston, MA, similar to the initial experiment's best configuration, we can see that the LSTM does well in general except for days with a large change in temperature which is to be expected.

### Test Accuracy of Model on 5-10 days' Collected After Model Trained
- Data used to initially train/test the model was from 1/1/2020 to 3/5/2025.
- To see how the model does on data collected after training, as of right now the **post-training weather data is from 3/6/2025 to 3/10/2025**. Please note, there is a delay in which this type of data (Global Historical Climatology Network - Daily (GHCN-Daily)) is available. For example I downloaded all the data on 3/13 and only got data up to 3/10.

- The model used for this is the best performing model from the second experiment. The exact configuration can be found in **LSTM_Recent_Data_Experiment.py**.

For March 3/6-3/10, the actual temperature high was 55.99999993, 42.9999996,  40.99999972, 45.99999942, and 62.00000005 respectively. 

The model predicted: 51.34314414, 45.10479152, 47.59140912, 47.08315473, 49.40315381 for each date. 

Thus, resulting in a **3.8236148922072273 test loss (MSE)**. The image below displays the actual temperature max vs the predicted temperature max.
![Boston Weather Prediction on data collected after model trained](images_for_readme\model_on_collected_data_posttraining.png)

