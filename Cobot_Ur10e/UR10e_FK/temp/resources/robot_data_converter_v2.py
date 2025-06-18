#!/usr/bin/env python3
"""
Robot Joint Data Converter

Converts CSV files containing robot joint data directly from numerical columns
to Excel format with organized columns for time, joint positions, TCP pose, and orientations.

Author: Generated for Robot Data Processing
"""

import pandas as pd
import numpy as np
import math
import os # os modülü eklendi

class RobotDataConverter:
    """Handles conversion of robot joint CSV data to Excel format."""
    
    def __init__(self):
        """Initialize the converter with joint mappings and column structure."""
        # Joint mapping is simplified as we now directly read q1-q6
        self.joint_order = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6'] 
        
        self.output_columns = [
            'time', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
            'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'qx', 'qy', 'qz', 'qw'
        ]
    
    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """
        Converts a quaternion to a 3x3 rotation matrix.
        
        Args:
            qx, qy, qz, qw (float): Quaternion components
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        
        # Normalize quaternion (ensure it's a unit quaternion)
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            return np.eye(3) # Return identity if quaternion is zero
        
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        # Calculate elements of the rotation matrix
        # (This uses the standard conversion from quaternion to rotation matrix)
        R11 = 1 - 2*(qy*qy + qz*qz)
        R12 = 2*(qx*qy - qz*qw)
        R13 = 2*(qx*qz + qy*qw)
        
        R21 = 2*(qx*qy + qz*qw)
        R22 = 1 - 2*(qx*qx + qz*qz)
        R23 = 2*(qy*qz - qx*qw)
        
        R31 = 2*(qx*qz - qy*qw)
        R32 = 2*(qy*qz + qx*qw)
        R33 = 1 - 2*(qx*qx + qy*qy)
        
        return np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])

    def matrix_to_rpy(self, T):
        """
        Converts a 4x4 transformation matrix to roll, pitch, and yaw (RPY) angles.
        Using ZYX Euler angle convention. (model.py'den kopyalanmıştır)
        """
        
        R = T[:3, :3]
        
        # Check for gimbal lock
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])  # roll
            y = math.atan2(-R[2,0], sy)     # pitch
            z = math.atan2(R[1,0], R[0,0])  # yaw
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return x, y, z
    
    def parse_single_row(self, line):
        """
        Parse a single CSV row and extract all relevant numerical data.
        
        Args:
            line (str): CSV data line
            
        Returns:
            dict: Parsed row data
        """
        line = line.strip()
        parts = line.split(',')
        
        try:
            # Convert all parts to float, handling potential errors
            float_parts = [float(p) for p in parts]
        except ValueError as e:
            print(f"Error parsing line as floats: {e} - Line: {line[:100]}...")
            # Return default/NaN values for this row if parsing fails
            return {col: 0.0 for col in self.output_columns if col != 'time'} # Exclude time, it's handled separately
            
        # Ensure enough data points are present
        if len(float_parts) < 14: # timestamp + 6 joints + 3 pos + 4 quaternion = 14
            print(f"Warning: Expected at least 14 numerical values, found {len(float_parts)} for line: {line[:100]}...")
            return {col: 0.0 for col in self.output_columns if col != 'time'} # Exclude time, it's handled separately
            
        # Initialize row data
        row_data = {}
        
        # Extract timestamp
        row_data['raw_timestamp'] = float_parts[0]
        
        # Extract joint positions (q1-q6)
        for i, q_name in enumerate(self.joint_order):
            row_data[q_name] = float_parts[1 + i] # Start from index 1 for q1
        
        # Extract TCP position (x, y, z)
        row_data['x'] = float_parts[7]
        row_data['y'] = float_parts[8]
        row_data['z'] = float_parts[9]
        
        # Extract orientation (quaternion)
        qx = float_parts[10]
        qy = float_parts[11]
        qz = float_parts[12]
        qw = float_parts[13]
        
        row_data['qx'] = qx
        row_data['qy'] = qy
        row_data['qz'] = qz
        row_data['qw'] = qw
        
        # Convert quaternion to rotation matrix
        rotation_matrix_3x3 = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Create a dummy 4x4 transformation matrix for matrix_to_rpy
        dummy_T = np.eye(4)
        dummy_T[:3, :3] = rotation_matrix_3x3
        
        # Convert to Euler angles using the matrix_to_rpy function from model.py's logic
        roll, pitch, yaw = self.matrix_to_rpy(dummy_T)
        row_data['roll'] = roll
        row_data['pitch'] = pitch
        row_data['yaw'] = yaw

        return row_data
    
    def read_csv_data(self, csv_path):
        """
        Read and parse the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            list: List of parsed row data
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

        print(f"Reading CSV file: {csv_path}")
        
        parsed_rows = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == 0: # Skip header line
                    continue
                if not line.strip(): # Skip empty lines
                    continue
                try:
                    row_data = self.parse_single_row(line)
                    # Sadece geçerli satırları ekle (dönüş değeri boş olmayanlar)
                    if row_data:
                        parsed_rows.append(row_data)
                except Exception as e:
                    print(f"Error processing row {i+1}: {e} - Line: {line[:100]}...")
                    continue
        
        print(f"Found {len(parsed_rows)} data rows")
        return parsed_rows
    
    def create_dataframe(self, parsed_rows):
        """
        Create a pandas DataFrame from parsed rows and process timestamps.
        
        Args:
            parsed_rows (list): List of parsed row dictionaries
            
        Returns:
            pd.DataFrame: Processed DataFrame with relative timestamps
        """
        # Create temporary DataFrame with raw timestamp
        temp_columns = ['raw_timestamp'] + [col for col in self.output_columns if col != 'time']
        df = pd.DataFrame(parsed_rows, columns=temp_columns)
        
        # Sort by timestamp (stable sort to preserve order for same timestamps)
        df = df.sort_values('raw_timestamp', kind='stable').reset_index(drop=True)
        
        # Create relative time starting from 0
        if len(df) > 1:
            min_timestamp = df['raw_timestamp'].min()
            df['time'] = (df['raw_timestamp'] - min_timestamp) # Zaman damgası farkı olarak hesapla
            # Zaman damgası saniye cinsinden olduğundan, bu haliyle kalsın.
        else:
            df['time'] = 1
        
        # Select and reorder final columns
        df = df[self.output_columns]
        
        return df
    
    def save_to_excel(self, df, excel_path):
        """
        Save DataFrame to Excel file.
        
        Args:
            df (pd.DataFrame): Data to save
            excel_path (str): Output Excel file path
        """
        print(f"Saving to Excel file: {excel_path}")
        try:
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"Successfully saved {len(df)} rows to {excel_path}")
        except Exception as e:
            print(f"Veri kaydedilirken hata oluştu: {e}")
            raise
    
    def print_summary(self, df):
        """
        Print conversion summary and statistics.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
        """
        print("\n" + "="*50)
        print("CONVERSION SUMMARY")
        print("="*50)
        print(f"Total rows processed: {len(df)}")
        
        if len(df) > 0:
            print(f"Time range: {df['time'].min():.1f} to {df['time'].max():.1f} seconds")
            print(f"Duration: {df['time'].max() - df['time'].min():.1f} seconds")
            
            print("\nJoint position ranges:")
            for col in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
                if col in df.columns:
                    print(f"   {col}: {df[col].min():.4f} to {df[col].max():.4f} rad")
            
            print("\nTCP position ranges:")
            for col in ['x', 'y', 'z']:
                if col in df.columns:
                    print(f"   {col}: {df[col].min():.4f} to {df[col].max():.4f} m")
            
            print(f"\nFirst few rows:")
            print(df.head(3).to_string())
        
        print("="*50)
    
    def convert(self, input_csv_path, output_excel_path):
        """
        Main conversion method.
        
        Args:
            input_csv_path (str): Path to input CSV file
            output_excel_path (str): Path to output Excel file
            
        Returns:
            pd.DataFrame: Converted DataFrame
        """
        try:
            # Read and parse CSV data
            parsed_rows = self.read_csv_data(input_csv_path)
            
            # Create and process DataFrame
            df = self.create_dataframe(parsed_rows)
            
            # Save to Excel
            self.save_to_excel(df, output_excel_path)
            
            # Print summary
            self.print_summary(df)
            
            return df
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise


def main():
    """Main function to run the conversion."""
    # Configuration
    # 'lasttest.csv' dosyasının projenizin kök dizininde olduğunu varsayılmıştır.
    input_csv = r"C:\Users\DELL\Documents\lasttest.csv" 
    output_excel = r"C:\Users\DELL\Documents\GitHub\model_based_digital_twin\Cobot_Ur10e\UR10e_FK\robot_joint_data_processed.xlsx" # Yeni bir çıktı dosyası ismi
    
    # Create converter and run conversion
    converter = RobotDataConverter()
    df = converter.convert(input_csv, output_excel)
    
    return df


if __name__ == "__main__":
    main()