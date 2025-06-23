import numpy as np
import pandas as pd
import gc
class DataQualityEnhancer:
    def __init__(self, memory_efficient=True, chunk_size=10000):
        self.column_types = {}
        self.categorical_encoders = {}
        self.feature_defs = None
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.scalers = {}
        
    def optimize_dtypes(self, df):
        optimized_df = df.copy()
        
        for column in optimized_df.columns:
            # Optimize integers
            if optimized_df[column].dtype.kind == 'i':
                if optimized_df[column].min() >= 0:
                    if optimized_df[column].max() < 255:
                        optimized_df[column] = optimized_df[column].astype(np.uint8)
                    elif optimized_df[column].max() < 65535:
                        optimized_df[column] = optimized_df[column].astype(np.uint16)
                    else:
                        optimized_df[column] = optimized_df[column].astype(np.uint32)
                else:
                    if optimized_df[column].min() > -128 and optimized_df[column].max() < 127:
                        optimized_df[column] = optimized_df[column].astype(np.int8)
                    elif optimized_df[column].min() > -32768 and optimized_df[column].max() < 32767:
                        optimized_df[column] = optimized_df[column].astype(np.int16)
                    else:
                        optimized_df[column] = optimized_df[column].astype(np.int32)
                        
            # Optimize floats
            elif optimized_df[column].dtype.kind == 'f':
                optimized_df[column] = optimized_df[column].astype(np.float32)
                
            # Optimize objects/strings
            elif optimized_df[column].dtype == 'object':
                if optimized_df[column].nunique() / len(optimized_df) < 0.5:  # If cardinality is low
                    optimized_df[column] = optimized_df[column].astype('category')
                    
        return optimized_df

    def _is_structured_text(self, series):
        """Check if text follows specific patterns (email, phone, etc.)"""
        # Sample a few non-null values
        sample = series.dropna().head(100)
        
        # Common patterns
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        phone_pattern = r'^\+?[\d\-\(\) ]{8,}$'
        url_pattern = r'^https?://[\w\.-]+\.\w+'
        
        if sample.empty:
            return False
            
        # Check if most values match any pattern
        pattern_matches = sample.astype(str).str.match(email_pattern) | \
                         sample.astype(str).str.match(phone_pattern) | \
                         sample.astype(str).str.match(url_pattern)
        
        return pattern_matches.mean() > 0.8

    def handle_missing_data(self, df):
        df_clean = df.copy()
        
        for column in df_clean.columns:
            # Ensure column type detection
            if column not in self.column_types:
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    self.column_types[column] = 'numerical'
                elif pd.api.types.is_string_dtype(df_clean[column]):
                    self.column_types[column] = 'categorical'
                elif pd.api.types.is_datetime64_any_dtype(df_clean[column]):
                    self.column_types[column] = 'datetime'
                else:
                    self.column_types[column] = 'text'
            
            # Handle missing values
            if df_clean[column].isnull().any():
                try:
                    if self.column_types[column] == 'numerical':
                        # Use median for all numerical columns
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                    
                    elif self.column_types[column] == 'categorical':
                        # Cardinality-based strategy
                        if df_clean[column].nunique() > 10:
                            df_clean[column] = df_clean[column].fillna('Missing')
                        else:
                            mode_val = df_clean[column].mode()
                            if len(mode_val) > 0:
                                df_clean[column] = df_clean[column].fillna(mode_val[0])
                            else:
                                df_clean[column] = df_clean[column].fillna('Unknown')
                    
                    elif self.column_types[column] == 'datetime':
                        # Custom datetime handling
                        df_clean = self._impute_datetime(df_clean, column)
                    
                    else:
                        # Default for text or undefined types
                        df_clean[column] = df_clean[column].fillna('Unknown')
                
                except Exception as e:
                    print(f"Warning: Issue with column '{column}'. Applying fallback. Error: {e}")
                    # Fallback to basic fillna
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                    else:
                        df_clean[column] = df_clean[column].fillna('Unknown')
        
        return df_clean
    
    def _impute_datetime(self, df, column):
        df_imputed = df.copy()
        
        # Convert to datetime if not already
        df_imputed[column] = pd.to_datetime(df_imputed[column], errors='coerce')
        
        # Forward fill with a limit
        df_imputed[column] = df_imputed[column].fillna(method='ffill', limit=3)
        
        # Backward fill remaining
        df_imputed[column] = df_imputed[column].fillna(method='bfill')
        
        # Any remaining NaTs get the median date
        median_date = df_imputed[column].median()
        df_imputed[column] = df_imputed[column].fillna(median_date)
        
        return df_imputed

    def perform_dfs(self, df, primitives_config=None):
        """
        Simplified feature engineering without featuretools for basic enhancement
        """
        feature_df = df.copy()
        
        # Basic feature engineering for numerical columns
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Create lag features if column has enough unique values
            if feature_df[col].nunique() > 10:
                feature_df[f'{col}_squared'] = feature_df[col] ** 2
                feature_df[f'{col}_log'] = np.log1p(np.abs(feature_df[col]))
        
        return feature_df
    
    def _chunked_dfs(self, df, primitives_config):
        """Process DFS in chunks for memory efficiency"""
        chunks = np.array_split(df, max(1, len(df) // self.chunk_size))
        feature_matrices = []
        
        for chunk in chunks:
            fm = self.perform_dfs(chunk, primitives_config)
            feature_matrices.append(fm)
            gc.collect()  # Force garbage collection
            
        return pd.concat(feature_matrices, axis=0)
    
    def remove_correlated_features(self, df, threshold=0.95):
        """
        Remove highly correlated features from the dataset
        """
        # Only consider numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return df
            
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr(method="pearson").abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop features
        df_uncorrelated = df.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} correlated features")
        return df_uncorrelated
        
    def standardize_and_encode(self, df):
        """
        Standardize numerical features and encode categorical ones
        """
        df_clean = df.copy()
        
        # Standardize numerical features
        numerical_cols = [col for col in df_clean.columns 
                         if self.column_types.get(col) == 'numerical' or 
                         df_clean[col].dtype in [np.int64, np.int32, np.float64, np.float32]]
        
        if numerical_cols:
            for col in numerical_cols:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                if std_val != 0:
                    df_clean[col] = (df_clean[col] - mean_val) / std_val
        
        # Encode categorical features
        categorical_cols = [col for col in df_clean.columns 
                           if self.column_types.get(col) in ['categorical', 'text'] or
                           df_clean[col].dtype == 'object']
        
        for col in categorical_cols:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                # Simple label encoding
                unique_vals = df_clean[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df_clean[col] = df_clean[col].map(mapping)
                self.categorical_encoders[col] = mapping
        
        return df_clean

    def detect_column_types(self, df):
        """
        Enhanced column type detection with column dropping
        """
        drop_columns = []
        
        for column in df.columns:
            # Sample data for large datasets
            if len(df) > 10000:
                sample = df[column].sample(n=min(10000, len(df)), random_state=42)
            else:
                sample = df[column]
                
            uniqueness_ratio = sample.nunique() / len(sample)
            null_ratio = sample.isnull().sum() / len(sample)
            
            # Drop columns with excessive nulls
            if null_ratio > 0.9:
                drop_columns.append(column)
                self.column_types[column] = 'drop_candidate'
                continue
            
            # Drop ID-like columns with very high uniqueness
            if uniqueness_ratio > 0.9 and len(sample) > 100:
                drop_columns.append(column)
                self.column_types[column] = 'id_drop_candidate'
                continue
        
        # Actually drop the identified columns
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True)
            print(f"Dropped {len(drop_columns)} columns: {drop_columns}")
        
        return self.column_types

    def enhance_data(self, df, primitives_config=None):
        """
        Main pipeline with additional safeguards and optimizations
        """
        try:
            print("Detecting and dropping unnecessary columns...")
            self.detect_column_types(df)
           
            print("Optimizing data types...")
            df_optimized = self.optimize_dtypes(df)
            
            print("Detecting column types...")
            self.detect_column_types(df_optimized)
            
            print("Handling missing data...")
            df_clean = self.handle_missing_data(df_optimized)
            
            print("Performing Feature Engineering...")
            df_features = self.perform_dfs(df_clean, primitives_config)
            
            print("Removing correlated features...")
            df_uncorrelated = self.remove_correlated_features(df_features)
            
            print("Performing final preprocessing...")
            df_final = self.standardize_and_encode(df_uncorrelated)
            
            if self.memory_efficient:
                gc.collect()
            
            return df_final
            
        except Exception as e:
            print(f"Error in data enhancement pipeline: {str(e)}")
            raise