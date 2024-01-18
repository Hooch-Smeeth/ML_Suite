import pandas as pd
import numpy as np
from math import isclose
import tensorflow as tf
from datastructures import EasyDict
from preprocess.prepare import normalize_columns


class TFTimeSeriesWindow():
  def __init__(self, data, input_width, label_width, batch_size, label_columns=None):
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(data.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.total_window_size = input_width + label_width

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
  
    dataset_dictionary = create_split_dataset_dict(data, sequence_length=self.total_window_size, batch_size=batch_size, data_type='timeseries')
    for name, dataset in dataset_dictionary.items():
      setattr(self, name, self.make_dataset(dataset))

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


  def make_dataset(self, data):
    data = data.map(self.split_window)
    return data


  def make_predictions(self, model):
      """
      Use the trained model to make predictions on the test set.

      Args:
          model (tf.keras.Model): The trained TensorFlow model.

      Returns:
          np.array: Predictions made by the model.
      """
      predictions = []
      for batch in self.test:  # Assuming 'self.test' is your test dataset
          inputs, _ = batch
          batch_predictions = model.predict(inputs)
          predictions.append(batch_predictions)

      # Flatten the list of predictions and return as a numpy array
      return np.concatenate(predictions, axis=0)


def create_split_dataset_dict(data, sequence_length, batch_size, data_type, 
                               split_ratio=[0.7, 0.2, 0.1], shuffle=True, framework="tensorflow"):
    """
    Splits the dataframe into train, validation, and test sets and converts them 
    into TensorFlow datasets based on the specified data type.

    Args:
        data (pd.DataFrame): The input DataFrame containing data.
        sequence_length (int): The length of the output sequences (in number of timesteps).
        batch_size (int): Number of samples in each batch.
        data_type (str): The type of data ('timeseries', 'nlp', 'image').
        split_ratio (List[float], optional): Ratio to split data into train, validation, and test sets.
        shuffle (bool, optional): Whether to shuffle the training dataset.
        framework (str, optional): The ML framework to use ('tensorflow', 'pytorch').

    Returns:
        An EasyDict object containing train, validation, and test TensorFlow datasets.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if not isinstance(split_ratio, list) or len(split_ratio) != 3:
        raise ValueError("Split ratio must be a list of three numbers.")

    if not isclose(sum(split_ratio), 1.0, rel_tol=1e-9):
        raise ValueError("Split ratios must sum to 1, within a small tolerance.")

    dataset_length = len(data)
    split1 = int(split_ratio[0] * dataset_length)
    split2 = int((split_ratio[0] + split_ratio[1]) * dataset_length)

    datasets = EasyDict(train = data[:split1],
                        val = data[split1:split2],
                        test = data[split2:])

    # Apply the normalization function to each subset in the dictionary
    for subset_name, subset_data in datasets.items():
        datasets[subset_name] = normalize_columns(subset_data, subset_data.columns)

    match framework:
      case 'tensorflow':
        match data_type:
            case 'timeseries' | 'nlp':
                for name, dataset in datasets.items():
                    dataset_np = np.array(dataset, dtype=np.float32)
                    datasets[name] = tf.keras.utils.timeseries_dataset_from_array(
                        data=dataset_np, targets=None, sequence_length=sequence_length,
                        sequence_stride=1, shuffle=shuffle if name == "train" else False,
                        batch_size=batch_size)
            case 'image':
                # Placeholder for image data processing and dataset creation
                pass
            case _:
                raise ValueError(f"Unsupported data type: {data_type}")
      case 'pytorch':
          # PyTorch specific dataset creation logic (placeholder)
          pass
      case _:
          raise ValueError(f"Unsupported framework: {framework}")

    return datasets
