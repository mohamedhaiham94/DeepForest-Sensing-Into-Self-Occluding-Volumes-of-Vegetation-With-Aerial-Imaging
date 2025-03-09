import os
import pickle
import numpy as np


class PKLFileHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_index = self._index_files()
        self.file_index2 = self._index_files2()

    def _index_files(self):
        """Indexes all .pkl files in the folder into a dictionary for efficient lookup."""
        file_index = {}
        
        # Get all .pkl files in the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pkl'):
                try:
                    # Extract the number1 and number2 from the filename
                    base_name = os.path.splitext(filename)[0]
                    number1, number2, number3 = map(int, base_name.split('_'))
                    
                    # Store the full path in the dictionary indexed by (number1, number2)
                    file_index[(number1, number2, number3)] = os.path.join(self.folder_path, filename)
                except ValueError:
                    # Skip files that don't match the expected format
                    continue

        return file_index
    
    def _index_files2(self):
        """Indexes all .pkl files in the folder into a dictionary for efficient lookup."""
        file_index = {}
        
        # Get all .pkl files in the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pkl'):
                try:
                    # Extract the number1 and number2 from the filename
                    base_name = os.path.splitext(filename)[0]
                    number1, number2 = map(int, base_name.split('_'))
                    
                    # Store the full path in the dictionary indexed by (number1, number2)
                    file_index[(number1, number2)] = os.path.join(self.folder_path, filename)
                except ValueError:
                    # Skip files that don't match the expected format
                    continue

        return file_index

    def get_file(self, number1, number2, number3):
        """Returns the file path of the corresponding .pkl file based on number1 and number2."""
        # Use tuple (number1, number2) to efficiently look up the file in the dictionary
        file_path = self.file_index.get((number1, number2, number3))
        if file_path:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data
        else:
            raise FileNotFoundError(f"No file found for {number1}_{number2}_{number3}.pkl")

    def get_file2(self, number1, number2):
        """Returns the file path of the corresponding .pkl file based on number1 and number2."""
        # Use tuple (number1, number2) to efficiently look up the file in the dictionary
        file_path = self.file_index2.get((number1, number2))
        if file_path:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data
        else:
            raise FileNotFoundError(f"No file found for {number1}_{number2}.pkl")



# if __name__ == '__main__':
#     # Usage example
#     fodler_path = r'd:\training_data\Layer_163\4x4x44'
#     pkl_handler = PKLFileHandler(fodler_path)

#     # Get the file based on number1 and number2
#     try:
#         data = pkl_handler.get_file(0, 336, 163)  # Example of number1=123 and number2=456
#         input_data = data['target']
#         print(len(input_data))
#         # np.set_printoptions(1)
#         print(np.array(input_data[21]))
#         # for i in np.array(input_data[21]):
#         #     print(i)
#         # print("Data loaded successfully.")
#     except FileNotFoundError as e:
#         print(e)
