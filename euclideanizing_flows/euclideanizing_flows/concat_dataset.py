import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import scipy.io as sio

class LASAConcatenated:
    def __init__(self, data_names):
        assert len(data_names) == 2, "This class expects exactly two data names."
        self.data_names = data_names
        self.x, self.xd, self.goal, self.dt, self.idx, self.scaling, self.translation = self.load_and_concatenate_data(data_names)

    def load_and_concatenate_data(self, data_names):
        # Placeholder paths and initialization
        dataset_path = '/Users/deniz/Documents/Research/dynamo/euclideanizing_flows/data/lasa_handwriting_dataset/'
        concatenated_x = []
        concatenated_xd = []
        idx = [0]
        goals = []
        scalings = []
        translations = []

        for name in data_names:
            data = sio.loadmat(os.path.join(dataset_path, name + '.mat'))
            dataset = data['demos'][0]  # Assuming 'demos' is directly accessible

            # Assuming the structure of your dataset allows direct extraction like this
            x, xd, current_idx, dt, goal, scaling, translation = self.process_dataset(dataset)
            concatenated_x.append(x)
            concatenated_xd.append(xd)
            goals.append(goal)
            scalings.append(scaling)
            translations.append(translation)
            idx.extend([i + idx[-1] for i in current_idx[1:]])

        # Concatenate all x and xd from both shapes
        final_x = np.concatenate(concatenated_x, axis=0)
        final_xd = np.concatenate(concatenated_xd, axis=0)
        final_goal = goals[-1]  # Assuming the goal of the last shape is the final goal
        final_scaling = np.mean(scalings, axis=0)  # Simplified scaling
        final_translation = np.mean(translations, axis=0)  # Simplified translation
        # Note: You might want to refine how scaling and translation are combined

        return final_x, final_xd, final_goal, dt, idx, final_scaling, final_translation

    def process_dataset(self, dataset):
        # Simplified processing, replace with your actual dataset processing logic
        # For demonstration, this just returns placeholders
        x = np.random.randn(150, 2)  # Placeholder
        xd = np.diff(x, axis=0, prepend=x[0:1])  # Placeholder
        idx = [0, 150]  # Placeholder
        dt = 0.01  # Placeholder
        goal = x[-1]  # Placeholder
        scaling = 1.0  # Placeholder
        translation = 0.0  # Placeholder
        return x, xd, idx, dt, goal, scaling, translation

    def plot_data(self):
        fig, ax = plt.subplots()
        for x in [self.x]:
            ax.scatter(x[:, 0], x[:, 1])
        plt.show()

# Example usage
data_names = ['Trapezoid', 'GShape']
lasa_conc = LASAConcatenated(data_names)
print("Goal:", lasa_conc.goal)
print('Dimensions:', np.shape(lasa_conc))
lasa_conc.plot_data()
