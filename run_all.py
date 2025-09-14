import subprocess, sys

def run(cmd):
    print(f"==> {cmd}")
    res = subprocess.run([sys.executable, cmd])
    if res.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")

def main():
    run("scripts/1_load_data.py")
    run("scripts/2_clean_rr.py")
    run("scripts/3_windowing.py")
    run("scripts/4_features_time.py")
    run("scripts/5_features_freq.py")
    # run("scripts/6_features_rsa.py")  # 可选
    run("scripts/7_make_table.py")
    # run("scripts/8_qc_plot.py")       # 可选
    print("Pipeline done.")

if __name__ == "__main__":
    main()