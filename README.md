# Deep Learning Charity Funding Predictor

## Project Overview

This project aims to create a binary classifier using deep learning techniques to predict the success of charitable organizations funded by Alphabet Soup. The goal is to achieve a predictive accuracy higher than 75% using neural network models.

## Table of Contents

1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Compilation, Training, and Evaluation](#model-compilation-training-and-evaluation)
4. [Model Optimization](#model-optimization)
5. [File Descriptions](#file-descriptions)
6. [Results](#results)

**Data Preprocessing**:
The data preprocessing step involves the following tasks:

    Reading the charity_data.csv file into a Pandas DataFrame.
    Identifying target and feature variables.
    Dropping unnecessary columns (EIN and NAME).
    Handling categorical variables:
        Determining the number of unique values for each column.
        Binning rare categorical variables.
        Encoding categorical variables using pd.get_dummies().
    Splitting the data into training and testing sets.
    Scaling the features using StandardScaler.

Example code for binning rare categorical variables:

python
```
# Choose a cutoff value and create a list of application types to be replaced
application_types_to_replace = list(application_count[application_count < 500].index)

# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

# Check to make sure replacement was successful
application_df['APPLICATION_TYPE'].value_counts()

Model Compilation, Training, and Evaluation
```

This step involves:

    Designing a neural network model using TensorFlow and Keras.
    Compiling and training the model.
    Evaluating the model's performance.

Look at CLASSIFICATION value counts to identify and replace with "Other"
python
```
classification_count = application_df['CLASSIFICATION'].value_counts()
classification_count
```

Example code for creating the neural network:

python
```
# Define the model - deep neural net
input_feature = len(X_train)
hidden_nodes_layer1 = 9
hidden_nodes_layer2 = 6

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=input_feature, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

**Model Optimization**:
To achieve the target accuracy of 75% or higher, various optimization techniques are employed:

    Adjusting input data (dropping columns, creating more bins, etc.).
    Modifying the neural network architecture (adding neurons, layers, etc.).
    Experimenting with different activation functions.
    Adjusting the number of epochs in training.

**File Descriptions**:

    charity_data.csv: The original dataset.
    AlphabetSoupCharity.ipynb: Jupyter notebook containing the initial model.
    AlphabetSoupCharity_Optimization.ipynb: Jupyter notebook with the optimized model.
    AlphabetSoupCharity.h5: HDF5 file containing the saved initial model.
    AlphabetSoupCharity_Optimization.h5: HDF5 file containing the saved optimized model.

## Results

### Performance of the Final Optimized Model

- **Accuracy**: The final optimized model achieved an accuracy of **72.85%** on the test data.
- **Loss**: The model’s final loss was **0.556**.

### Insights from the Optimization Process
- **Initial Performance**: The initial model without any optimizations performed well but did not meet the desired accuracy threshold of 75%. Its accuracy plateaued at around 73% after 100 epochs.
- **Optimization Techniques**: 
  - **Adjusting Input Data**: By dropping columns `EIN` and `NAME`, it helped remove non-contributing features that helped reduce noise in the data.
  - **Neurons and Layers**: Increasing the number of neurons in the hidden layers and experimenting with additional hidden layers helped refine the learning process, but adding too many neurons resulted in overfitting.
  - **Activation Functions**: Using `ReLU` for hidden layers and `sigmoid` for the output layer provided a balance between complexity and convergence. 
  - **Epochs**: The model trained for 100 epochs, which provided sufficient training time without overfitting.
    
Overall, while the final accuracy of 72.85% is below the target of 75%, the improvements in performance from initial to final models show that the network can reasonably predict the success of organizations. Further optimization or more complex models might be needed to reach the 75% threshold.
