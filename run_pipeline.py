from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    pipeline_instance = training_pipeline(data_path="dataset/olist_customers_dataset.csv")