import pandas as pd
import os


def clean_dataset(file_path, output_dir):
    # Load
    df = pd.read_csv(file_path)

    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["reviews.rating", "reviews.text"], inplace=True)
    df["reviews.text"] = df["reviews.text"].str.lower().str.strip()

    # Optional: Convert date columns to datetime
    df["reviews.date"] = pd.to_datetime(df["reviews.date"], errors="coerce")

    # Save to Parquet
    base = os.path.basename(file_path).split(".")[0]
    output_path = os.path.join(output_dir, f"{base}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved cleaned file to: {output_path}")

# Example usage
if __name__ == "__main__":
    clean_dataset(r"Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_3.csv", 
                  r"\Scalable Hybrid Recommendation System on Amazon Product Reviews\data")



import pandas as pd
import glob

def combine_parquets(parquet_folder):
    all_files = glob.glob(f"{parquet_folder}/*.parquet")
    df_list = [pd.read_parquet(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# Example usage
if __name__ == "__main__":
    df = combine_parquets("path")
    print("✅ Combined DataFrame shape:", df.shape)
    df.to_parquet("path\combined.parquet", index=False)
