from requests import get
from infra.db.database_utils import DatabaseEngine, DatabaseRepository, DatabaseFactory, get_postgres_db_engine
import pandas as pd
import argparse

argparser = argparse.ArgumentParser(description="Insert Kaggle training data into the database.")
argparser.add_argument("--kaggle_train_data_path", type=str, required=True, help="Path to the Kaggle training data CSV file.")
args = argparser.parse_args()

database_engine = get_postgres_db_engine()

def save_kaggle_train_data_to_db(kaggle_train_data_path: str, db_engine: DatabaseEngine) -> None:
    """
    Saves the Kaggle training data to the database.
    
    Args:
        kaggle_train_data: DataFrame containing Kaggle training data.
        db_engine: Database engine instance for database operations.
    """
    repository = DatabaseRepository(engine=db_engine)
    kaggle_train_data = pd.read_csv(kaggle_train_data_path)
    
    print(f"Kaggle data shape: {kaggle_train_data.shape}")
    print(f"Number of columns: {len(kaggle_train_data.columns)}")
    print(f"Columns: {kaggle_train_data.columns.tolist()[:10]}...")  # Show first 10
    
    try:
        # Use chunksize to avoid issues with large inserts
        repository.save_dataframe(kaggle_train_data, table_name="kaggle_train_data", if_exists='replace')
        print(f"Successfully saved {len(kaggle_train_data)} rows to database")
    except Exception as e:
        print(f"Error saving Kaggle training data to database: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        repository.close()
        
save_kaggle_train_data_to_db(args.kaggle_train_data_path, database_engine)