import unittest
import os
import random
import sys
import yaml
from unittest.mock import patch

# Import your main script functions
from src.core.generator import generate_clean_telemetry_data
from src.core.vehicles import get_vehicle_classes
from src.core.anomalies import get_anomaly_classes
from src.core.utils import introduce_anomalies, create_file_csv


class TestSyntheticDatasetGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test directories and parameters with Docker volume paths."""
        # Get the project root directory (where data folder should be)
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.data_dir = os.path.join(cls.project_root, 'data')

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        cls.vehicle_type = config['vehicle']['type']
        cls.initial_speed = config['vehicle']['initial_speed']
        cls.minutes = config['timeseries']['minutes']
        cls.anomaly_type = config['anomaly']['type']
        cls.anomaly_duration = config['anomaly']['duration']
        cls.anomaly_probability = config['anomaly']['probability']

        # Define test directory structure inside the data folder
        cls.data_subdirs = {
            'training': os.path.join(cls.data_dir, "training"),
            'validation': os.path.join(cls.data_dir, "validation"),
            'testing': os.path.join(cls.data_dir, "testing")
        }
        
        cls.vehicle_arg = f"{cls.vehicle_type}:{cls.initial_speed}"
        cls.anomalies = [f"{cls.anomaly_type}:{cls.anomaly_duration}:{cls.anomaly_probability}"]
        
        # Define test file paths
        cls.test_files = {
            'training': os.path.join(cls.data_subdirs['training'], f"train_{cls.vehicle_arg}_{cls.minutes}.csv"),
            'validation': os.path.join(cls.data_subdirs['validation'], f"val_{cls.vehicle_arg}_{cls.minutes}.csv"),
            'testing': os.path.join(cls.data_subdirs['testing'], 
                                    f"test_{cls.vehicle_arg}_{cls.minutes}_{' '.join(cls.anomalies)}.csv")
        }
        
        # Create directories if they don't exist
        for dir_path in cls.data_subdirs.values():
            os.makedirs(dir_path, exist_ok=True)


    def test_directory_structure_created(self):
        """Test that required directories exist."""
        # Verify data directory exists at the correct location
        self.assertTrue(os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist")
        
        # Verify subdirectories exist
        for dir_type, dir_path in self.data_subdirs.items():
            self.assertTrue(os.path.exists(dir_path), f"{dir_type} directory {dir_path} does not exist")
            self.assertTrue(os.path.isdir(dir_path), f"Path {dir_path} is not a directory")

    def generate_dataset(self, output_file, anomalies=None, seed=42):
        """Helper function to generate a dataset with specific parameters."""
        test_args = [
            "--vehicle", self.vehicle_arg,
            "--minutes", str(self.minutes),
            "--output", output_file
        ]
        
        # Add anomalies if specified (for testing set)
        if anomalies:
            test_args.extend(["--anomalies", *anomalies, "--labels"])
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Patch sys.argv to simulate command line arguments
        with patch.object(sys, 'argv', ['test_script.py'] + test_args):
            from src.generate_dataset import main
            main()

    def test_all_datasets_generation(self):
        """Test that all three datasets are generated correctly."""
        # Generate training dataset (no anomalies)
        self.generate_dataset(self.test_files['training'], seed=42)
        
        # Generate validation dataset (no anomalies)
        self.generate_dataset(self.test_files['validation'], seed=43)
        
        # Generate testing dataset (with anomalies)
        self.generate_dataset(self.test_files['testing'], anomalies=self.anomalies, seed=44)
        
        # Verify all files were created
        for dataset_type, file_path in self.test_files.items():
            with self.subTest(dataset_type=dataset_type):
                self.assertTrue(os.path.exists(file_path), 
                              f"{dataset_type} dataset was not created at {file_path}")
                
                # Verify file content
                with open(file_path, 'r') as f:
                    content = f.read()
                    self.assertGreater(len(content), 0, "File is empty")
                    
                    # Check if it's a proper CSV
                    lines = content.split('\n')
                    self.assertGreater(len(lines), 1, "File should have header and data")
                    
                    # For testing set, verify it has anomaly labels if specified
                    if dataset_type == 'testing':
                        self.assertIn('anomaly', lines[0].lower(), 
                                    "Testing set should have anomaly labels")

    def test_dataset_differences(self):
        """Test that datasets are different due to different seeds."""
        # Generate all datasets
        self.test_all_datasets_generation()
        
        # Collect file contents
        file_contents = {}
        for name, path in self.test_files.items():
            with open(path, 'r') as f:
                file_contents[name] = f.read()
        
        # Verify all datasets are different
        self.assertNotEqual(file_contents['training'], file_contents['validation'],
                          "Training and validation datasets should be different")
        self.assertNotEqual(file_contents['validation'], file_contents['testing'],
                          "Validation and testing datasets should be different")
        self.assertNotEqual(file_contents['training'], file_contents['testing'],
                          "Training and testing datasets should be different")


if __name__ == '__main__':
    unittest.main()