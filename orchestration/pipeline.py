import subprocess

def run_pipeline():
    print("Running Silver pipeline...")
    subprocess.run(["python", "pipelines/silver.py"], check=True)

    print("Running Gold pipeline...")
    subprocess.run(["python", "pipelines/gold.py"], check=True)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()